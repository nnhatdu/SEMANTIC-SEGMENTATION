
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
slim=tf.contrib.slim
from utils import preprocessing
_BATH_NORM_DECAY=0.9997
_WEIGHT_DECAY=5e-4


# In[10]:


def model_fn(features,labels,mode,params):
    '''Model interface for estimator
     parameter:
       features: input features
       labels: true label
       mode: model mode
       params: parameters related to model operation
     return value:
       Model interface form
      '''
    if isinstance(features,dict):
        features=features['feature']
    images=tf.cast(tf.map_fn(preprocessing.mean_image_addition,features),
                  tf.uint8)
    network=model_generator(params['num_classes'],
                           params['output_stride'],
                           params['base_architecture'],
                           params['pre_trained_model'],
                           params['batch_norm_decay'])
    logits=network(features,mode==tf.estimator.ModeKeys.TRAIN)
    pred_classes=tf.expand_dims(tf.argmax(logits,axis=3,output_type=tf.int32),axis=3)
    pred_decoded_labels=tf.py_func(preprocessing.decode_labels,
                                  [pred_classes,params['batch_size'],params['num_classes']],
                                  tf.uint8)
    
    predictions={
        'classes':pred_classes,
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor'),
        'decoded_labels':pred_decoded_labels
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        predictions_without_decoded_labels=predictions.copy()
        del predictions_without_decoded_labels['decoded_labels']
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'preds':tf.estimator.export.PredictOutput(
                    predictions_without_decoded_labels)
                })
    gt_decoded_labels=tf.py_func(preprocessing.decode_labels,
                                [labels,params['batch_size'],params['num_classes']],tf.uint8)
    
    labels=tf.squeeze(labels,axis=3)#[batch,h,w]
    logits_by_num_classes=tf.reshape(logits,[-1,params['num_classes']])#[-1,21]
    labels_flat=tf.reshape(labels,[-1,])#[-1]
    valid_indices=tf.to_int32(labels_flat<=params['num_classes']-1)
    valid_logits=tf.dynamic_partition(logits_by_num_classes,valid_indices,num_partitions=2)[1]#[-1,num_classes]
    valid_labels=tf.dynamic_partition(labels_flat,valid_indices,num_partitions=2)[1]#[-1]
    
    pred_flat=tf.reshape(pred_classes,[-1,])#[-1]
    valid_preds=tf.dynamic_partition(pred_flat,valid_indices,num_partitions=2)[1]#[-1]
    confusion_matrix=tf.confusion_matrix(valid_labels,valid_preds,num_classes=params['num_classes'])
    predictions['valid_preds']=valid_preds
    predictions['valid_labels']=valid_labels
    predictions['confusion_maxtrix']=confusion_matrix
    
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(
    logits=valid_logits,labels=valid_labels)
    
    tf.identity(cross_entropy,name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
    if not params['freeze_batch_norm']:
        train_var_list=[v for v in tf.trainable_variables()]
    else:
        train_var_list=[v for v in tf.trainable_variables()
                       if 'beta' not in v.name and 'gamma' not in v.name]
        loss=cross_entropy+params.get('weight_decay',_WEIGHT_DECAY)*tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
        
    if mode==tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('image',
                        tf.concat(axis=2,values=[images,gt_decoded_labels,pred_decoded_labels]),
                        max_outputs=params['tensorboard_images_max_outputs'])
        global_step=tf.train.get_or_create_global_step()
        if params['learning_rate_policy']=='piecewise':
            initial_learning_rate=0.1*params['batch_size']/128
            batches_per_epoch=params['num_train']/params['batch_size']
            boundaries=[int(batches_per_epoch*epoch) for epoch in [100,150,200]]
            values=[initial_learning_rate*decay for decay in [1,0.1,0.01,0.001]]
            learning_rate=tf.train.piecewise_constant(
            tf.cast(global_step,tf.int32),boundaries,values)
        elif params['learning_rate_policy']=='poly':
            learning_rate=tf.train.polynomial_decay(
            params['initial_learning_rate'],
            tf.cast(global_step,tf.int32)-params['initial_global_step'],
            params['max_iter'],params['end_learning_rate'],power=params['power'])
        else:
            raise ValueError('Choose a learning rate model')
        tf.identity(learning_rate,name='learning_rate')
        tf.summary.scalar('learning_rate',learning_rate)
        
        tf.identity(global_step,name='global_step')
        tf.summary.scalar('global_step',global_step)
        optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=params['momentum'])
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op=optimizer.minimize(loss,global_step,var_list=train_var_list)
    else:
        train_op=None
    
    accuracy=tf.metrics.accuracy(valid_labels,valid_preds)
    mean_iou=tf.metrics.mean_iou(valid_labels,valid_preds,params['num_classes'])
    metrics={'px_accuracy':accuracy,'mean_iou':mean_iou}
    
    tf.identity(accuracy[1],name='train_px_accuracy')
    tf.summary.scalar('train_px_accuracy',accuracy[1])
    
    def compute_mean_iou(total_cm,name='mean_iou'):
        '''Calculate the average iou
         parameter:
           total_cm: confusion matrix
         Return value: mean iou
        '''
        sum_over_row=tf.to_float(tf.reduce_sum(total_cm,0))
        sum_over_col=tf.to_float(tf.reduce_sum(total_cm,1))
        cm_diag=tf.to_float(tf.diag_part(total_cm))
        denominator=sum_over_row+sum_over_col-cm_diag
        
        num_valid_entries=tf.reduce_sum(tf.cast(
            tf.not_equal(denominator,0),dtype=tf.float32))
        denominator=tf.where(tf.greater(
            denominator,0),denominator,
            tf.ones_like(denominator))
        iou=tf.div(cm_diag,denominator)
        
        for i in range(params['num_classes']):
            tf.identity(iou[i],name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i),iou[i])
        result=tf.where(
            tf.greater(num_valid_entries,0),
            tf.reduce_sum(iou,name=name)/num_valid_entries,
            0)
        return result
    train_mean_iou=compute_mean_iou(mean_iou[1])
    tf.identity(train_mean_iou,name='train_mean_iou')
    tf.summary.scalar('train_mean_iou',train_mean_iou)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


# In[1]:


def model_generator(num_classes,output_stride,
                   base_architecture,
                   pre_trained_model,
                   batch_norm_decay,
                   data_format='channels_last'):
    '''Model main program
     parameter:
       num_classes: categories
       output_stride: The step size of resnet is also related to the hole convolution expansion coefficient. If it is 16, the coefficient is [6, 12, 18], and if it is 8, the coefficient is doubled
       base_architecture: overloaded model for resnet
       pre_trained_model: Pre-trained model directory
       batch_norm_decay: the coefficients of the BN layer
       data_format: The format of the input image, whether the RGB channel is the first or the last
     return value:
       Return the predicted value shape[batch,h,w,num_classes]
      '''
    if data_format is None:
        pass
    if batch_norm_decay is None:
        batch_norm_decay=_BATH_NORM_DECAY
    if base_architecture not in ['resnet_v2_50','resnet_v2_101']:
        raise ValueError('The overloaded model is not correct')
    if base_architecture =='resnet_v2_50':
        base_model=resnet_v2.resnet_v2_50
    else:
        base_model=resnet_v2.resnet_v2_101
    def model(inputs,is_training):
        if data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])
       
        with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits,end_points=base_model(inputs,
                                        num_classes=None,
                                        is_training=is_training,
                                        global_pool=False,
                                        output_stride=output_stride)
        if is_training:
            exclude=[base_architecture+'/logits','global_step']
            variables_to_restore=slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(pre_trained_model,
                                         {v.name.split(':')[0]: v for v in variables_to_restore})
        inputs_size=tf.shape(inputs)[1:3]
        net=end_points[base_architecture+'/block4']
        encoder_output=atrous_spatial_pyramid_pooling(net,output_stride,batch_norm_decay,is_training)
        
        with tf.variable_scope('decoder'):
            with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
#                 with slim.arg_scope([slim.conv2d],
#                                     weights_initializer=slim.xavier_initializer(),

#                                     normalizer_fn=slim.batch_norm,
#                                     normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
                with tf.variable_scope('low_level_features'):
                    low_level_features=end_points[base_architecture+'/block1/unit_3/bottleneck_v2/conv1']
                    low_level_features=slim.conv2d(low_level_features,48,[1,1],stride=1,scope='conv_1x1')
                    low_level_features_size=tf.shape(low_level_features)[1:3]

                with tf.variable_scope('upsampling_logits'):
                    net=tf.image.resize_bilinear(encoder_output,low_level_features_size,name='upsample_1')
                    net=tf.concat([net,low_level_features],axis=3,name='concat')
                    net=slim.conv2d(net,256,[3,3],stride=1,scope='conv_3x3_1')
                    net=slim.conv2d(net,256,[3,3],stride=1,scope='conv_3x3_2')
                    net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='conv_1x1')
                    logits=tf.image.resize_bilinear(net,inputs_size,name='upsample_2')
        return logits
    return model


# In[7]:


def atrous_spatial_pyramid_pooling(inputs,output_stride,
                                  batch_norm_decay,is_training,depth=256):
    '''Implement ASPP
     parameter:
       inputs: input four-dimensional vector
       output_stride: Determines the dilation rate of the dilated convolution
       batch_norm_decay: same function as above
       is_training: whether to train
       depth: number of output channels
     return value:
       Output after ASPP
      '''
    with tf.variable_scope('aspp'):
        if output_stride not in [8,16]:
            raise ValueError('incorrect out_stride')
        atrous_rates=[6,12,18]
        if output_stride ==8:
            atrous_rates=[2*rate for rate in atrous_rates]
        with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.xavier_initializer(),
                                
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
                inputs_size=tf.shape(inputs)[1:3]
                conv_1x1=slim.conv2d(inputs,depth,[1,1],stride=1,scope='conv_1x1')
                conv_3x3_1=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[0],scope='conv_3x3_1')
                conv_3x3_2=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[1],scope='conv_3x3_2')
                conv_3x3_3=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[2],scope='conv_3x3_3')
                with tf.variable_scope('image_level_features'):
                    image_level_features=tf.reduce_mean(inputs,axis=[1,2],keep_dims=True,name='global_average_pooling')
                    image_level_features=slim.conv2d(image_level_features,depth,[1,1],stride=1,scope='conv_1x1')
                    image_level_features=tf.image.resize_bilinear(image_level_features,inputs_size,name='upsample')
                net=tf.concat([conv_1x1,conv_3x3_1,conv_3x3_2,conv_3x3_3,image_level_features],axis=3,name='concat')
                return net

