
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from utils import   deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug
from utils import config as FLAGS
import shutil
_NUM_CLASSES = 21
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}

# In[ ]:


def main():
    session_config = tf.ConfigProto(device_count={'GPU': 0,'GPU':1})



    run_config=tf.estimator.RunConfig().replace(session_config=session_config,save_checkpoints_secs=1e2, keep_checkpoint_max = 3)
    
    model=tf.estimator.Estimator(model_fn=deeplab_model.model_fn,
                                 model_dir=FLAGS.model_dir,
                                 config=run_config,
                                 params={
                                     'output_stride': FLAGS.output_stride,
                                      'batch_size': FLAGS.batch_size,
                                      'base_architecture': FLAGS.base_architecture,
                                      'pre_trained_model': FLAGS.pre_trained_model,
                                      'batch_norm_decay': _BATCH_NORM_DECAY,
                                      'num_classes': _NUM_CLASSES,
                                      'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
                                      'weight_decay': FLAGS.weight_decay,
                                      'learning_rate_policy': FLAGS.learning_rate_policy,
                                      'num_train': _NUM_IMAGES['train'],
                                      'initial_learning_rate': FLAGS.initial_learning_rate,
                                      'max_iter': FLAGS.max_iter,
                                      'end_learning_rate': FLAGS.end_learning_rate,
                                      'power': _POWER,
                                      'momentum': _MOMENTUM,
                                      'freeze_batch_norm': FLAGS.freeze_batch_norm,
                                      'initial_global_step': FLAGS.initial_global_step
                                 })
    for _ in range(FLAGS.train_epochs//FLAGS.epochs_per_eval):
        tensors_to_log={
              'global_step':'global_step',
             'learning_rate': 'learning_rate',
              'cross_entropy': 'cross_entropy',
              'train_px_accuracy': 'train_px_accuracy',
              'train_mean_iou': 'train_mean_iou',
               
            }
        loggig_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=10)
        train_hooks=[loggig_hook]
        eval_hooks=None

        if FLAGS.debug:
            debug_hook=tf_debug.LocalCLIDebugHook()
            train_hooks.append(debug_hook)
            eval_hooks=[debug_hook]
        tf.logging.info('Start training')
        model.train(input_fn=lambda:input_fn(True,FLAGS.tfrecord_path,FLAGS.batch_size,FLAGS.epochs_per_eval),
                   hooks=train_hooks)
        tf.logging.info('Start validating')
        eval_results=model.evaluate(
            input_fn=lambda : input_fn(False,FLAGS.tfrecord_path,1),
            hooks=eval_hooks)
        print(eval_results)


# In[ ]:


def input_fn(is_training,data_dir,batch_size,num_epochs=1):
    '''Convert data into estimator input format'''
    dataset=tf.data.Dataset.from_tensor_slices(get_filenames(is_training,data_dir))
    dataset=dataset.flat_map(tf.data.TFRecordDataset)
    if is_training:
        dataset=dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    dataset=dataset.map(parse_record)
    dataset=dataset.map(
        lambda image,label: preprocess_image(image,label,is_training))
    dataset=dataset.prefetch(batch_size)
    dataset=dataset.repeat(num_epochs)
    dataset=dataset.batch(batch_size)
    
    iterator=dataset.make_one_shot_iterator()
    images,labels=iterator.get_next()
    return images,labels


# In[ ]:


def get_filenames(is_training,data_dir):
    '''get data directory'''
    if is_training:
        return [os.path.join(data_dir,'train.record')]
    else:
        return [os.path.join(data_dir,'val.record')]


# In[ ]:


def parse_record(raw_record):
    '''Parse tfrecord data'''
    key_to_features={
        'image':tf.FixedLenFeature((),tf.string,default_value=''),
        'label':tf.FixedLenFeature((),tf.string,default_value='')
    }
    parsed=tf.parse_single_example(raw_record,key_to_features)
    image=tf.image.decode_image(
        tf.reshape(parsed['image'],shape=[]),_DEPTH)
    image=tf.to_float(tf.image.convert_image_dtype(image,dtype=tf.uint8))
    image.set_shape([None,None,3])
    
    label=tf.image.decode_image(
        tf.reshape(parsed['label'],shape=[]),1)
    label=tf.to_int32(tf.image.convert_image_dtype(label,dtype=tf.uint8))
    label.set_shape([None,None,1])
    return image,label


# In[ ]:


def preprocess_image(image,label,is_training):
    '''data preprocessing'''
    if is_training:
        image,label=preprocessing.random_rescale_image_and_label(
            image,label,_MIN_SCALE,_MAX_SCALE)
        image,label=preprocessing.random_crop_or_pad_image_and_label(
            image,label,_HEIGHT,_WIDTH,_IGNORE_LABEL)
        image,label=preprocessing.random_filp_left_right_image_and_label(
            image,label)
        image.set_shape([_HEIGHT,_WIDTH,3])
        label.set_shape([_HEIGHT,_WIDTH,1])
    image=preprocessing.mean_image_subtraction(image)
    return image,label


# In[ ]:


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

