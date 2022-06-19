
# coding: utf-8

# In[1]:


''' Mainly perform related data preprocessing'''
from PIL import  Image
import numpy as np
import tensorflow as tf

_R_MEAN=123.68
_G_MEAN=116.78
_B_MEAN=103.94

label_colors=[(0,0,0),#0=background
              #1=airplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
              (128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
              #6=Bus, 7=Car, 8=Cat, 9=Chair, 10=Cow
              (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),
              #11=dinner table, 12=dog, 13=horse, 14=motorcycle, 15=person
              (192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
              #16=pot, 17=sheep, 18=sofa, 19=train, 20=television or monitor
              (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]


# In[3]:


def decode_labels(mask,num_image=1,num_classes=21):
    '''color the picture
       parameter:
         mask: shape is the category of [batch, h, w, 1] pixel value of each pixel
         num_image: The length of the image to be processed each time
         num_classes: number of classification categories
       return value:
         Returns the colored segmented image
        '''
    n,h,w,c=mask.shape
    assert (n>=num_image),'num_image %d Cannot compare batches %d heavy'                            %(n,num_image)
    outputs=np.zeros((num_image,h,w,3),dtype=np.uint8)
    for i in range(num_image):
        img=Image.new('RGB',(len(mask[i,0]),len(mask[i])))
        pixels=img.load()
        for j_,j in enumerate(mask[i,:,:,0]):
            for k_,k in enumerate(j):
                if k<num_classes:
                    pixels[k_,j_]=label_colors[k]
        outputs[i]=np.array(img)
    return outputs


# In[4]:


def mean_image_addition(image,means=(_R_MEAN,_G_MEAN,_B_MEAN)):
    '''Add the mean value for each channel of the image
     parameter:
       image: mean-subtracted image shape[h,w,c]
       means: the mean of each channel
     Return value: the image after adding the mean
    '''
    if image.get_shape().ndims!=3:
        raise ValueError('invalid image')
    num_channels=image.get_shape().as_list()[-1]
    if len(means)!=num_channels:
        raise ValueError('invalid mean')
    channels=tf.split(axis=2,num_or_size_splits=num_channels,value=image)
    for i in range(num_channels):
        channels[i]+=means[i]
    return tf.concat(axis=2,values=channels)
      


# In[6]:


def mean_image_subtraction(image,means=(_R_MEAN,_G_MEAN,_B_MEAN)):
    '''Image minus mean as input
     parameter:
       image: original image [h,w,c]
       means: mean
     return value:
       The mean subtracted image is used as input
    '''
    if image.get_shape().ndims!=3:
        raise ValueError('invalid image')
    num_channels=image.get_shape().as_list()[-1]
    if len(means)!=num_channels:
        raise ValueError('invalid mean')
    channels=tf.split(axis=2,num_or_size_splits=num_channels,value=image)
    for i in range(num_channels):
        channels[i]-=means[i]
    return tf.concat(axis=2,values=channels)


# In[8]:


def random_rescale_image_and_label(image,label,min_scale,max_scale):
    '''Randomly zoom in and out of images
     parameter:
       image: input image [h,w,c]
       label: segmented output image [h,w,1]
       min_scale, max_scale: scale changes the minimum and maximum values
     return value:
       Change the scale of the image and label
      '''
    if min_scale<=0:
        raise ValueError('The minimum scale must be greater than 0')
    elif max_scale<=0:
        raise ValueError('The maximum scale must be greater than 0')
    elif min_scale>=max_scale:
        raise ValueError('wrong size')
    shape=tf.shape(image)
    height=tf.to_float(shape[0])
    width=tf.to_float(shape[1])
    scale=tf.random_uniform([],minval=min_scale,maxval=max_scale,dtype=tf.float32)
    
    new_height=tf.to_int32(height*scale)
    new_width=tf.to_int32(width*scale)
    image=tf.image.resize_images(image,[new_height,new_width],
                                method=tf.image.ResizeMethod.BILINEAR)
    label=tf.image.resize_images(label,[new_height,new_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image,label


# In[9]:


def random_crop_or_pad_image_and_label(image,label,crop_height,crop_width,ignore_label):
    '''Random crop to fill the image
     parameter:
       image: input image [h,w,c]
       label: output label[h,w,1]
       crop_height, crop_width: new image size
       ignore_label: the category to be ignored
     return value:
        The processed image, label
       '''
    label=label-ignore_label
    label=tf.to_float(label)
    shape=tf.shape(image)
    height=shape[0]
    width=shape[1]
    image_and_label=tf.concat([image,label],axis=2)
    image_and_label_pad=tf.image.pad_to_bounding_box(
            image_and_label,0,0,
            tf.maximum(crop_height,height),
            tf.maximum(crop_width,width))
    image_and_label_crop=tf.random_crop(
        image_and_label_pad,[crop_height,crop_width,4])
    image_crop=image_and_label_crop[:,:,:3]
    label_crop=image_and_label_crop[:,:,3:]
    label_crop+=ignore_label
    label_crop=tf.to_int32(label_crop)
    return image_crop,label_crop


# In[10]:


def random_filp_left_right_image_and_label(image,label):
    '''Randomly flip the image left and right
     parameter:
       image: input image [h,w,c]
       label: output label[h,w,1]
     return value:
       The processed image, label
      '''
    uniform_random=tf.random_uniform([],0,1.0)
    mirror_cond=tf.less(uniform_random,0.5)
    image=tf.cond(mirror_cond,lambda: tf.reverse(image,[1]),lambda:image)
    label=tf.cond(mirror_cond,lambda:tf.reverse(label,[1]),lambda:label)
    return image,label
    


# In[11]:


def eval_input_fn(image_filenames,label_filenames=None,batch_size=1):
    '''Process the image folder into a model receiving data format
     parameter:
       image_filenames: image directory
       label_filenames: The test data has no label
       Put batch_size: test default batch is 1
     return value:
       The data in the form of data contains image and label
      '''
    def _parse_function(filename,is_label):
        if not is_label:
            image_filename,label_filename=filename,None
        else :
            image_filename,label_filename=filename
        image_string=tf.read_file(image_filename)
        image=tf.image.decode_image(image_string)
        image=tf.to_float(tf.image.convert_image_dtype(image,dtype=tf.uint8))
        image.set_shape([None,None,3])
        image=mean_image_subtraction(image)
        if not is_label:
            return image
        else:
            label_string = tf.read_file(label_filename)
            label = tf.image.decode_image(label_string)
            label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
            label.set_shape([None, None, 1])
            return image,label
    if label_filenames is None:
        input_filenames=image_filenames
    else:
        input_filenames=(image_filenames,label_filenames)
    dataset=tf.data.Dataset.from_tensor_slices(input_filenames)
    if label_filenames is None:
        dataset=dataset.map(lambda x: _parse_function(x,False))
    else:
        dataset=dataset.map(lambda x,y:_parse_function((x,y),True))
    dataset=dataset.prefetch(batch_size)
    dataset=dataset.batch(batch_size)
    iterator=dataset.make_one_shot_iterator()
    if label_filenames is None:
        images=iterator.get_next()
        labels=None
    else:
        images,labels=iterator.get_next()
    return images,labels