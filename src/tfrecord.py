
# coding: utf-8

# In[1]:


import io
import os
from PIL import Image
import tensorflow as tf
from utils import config as Config
from utils import dataset_util


# In[2]:


def main():
    '''Generate tfrecords main program'''
    if not os.path.exists(Config.tfrecord_path):
        os.makedirs(Config.tfrecord_path)
    tf.logging.info('read data')
    
    image_dir=os.path.join(Config.data_dir,Config.image_data_dir)
    label_dir=os.path.join(Config.data_dir,Config.label_data_dir)
    
    if not os.path.isdir(label_dir):
        raise ValueError('Missing data, go to download')
    train_examples=dataset_util.read_examples_list(Config.train_data_list)
    val_examples=dataset_util.read_examples_list(Config.val_data_list)
    
    train_output_path=os.path.join(Config.tfrecord_path,'train.record')
    val_output_path=os.path.join(Config.tfrecord_path,'val.record')
    
    create_record(train_output_path,image_dir,label_dir,train_examples)
    create_record(val_output_path,image_dir,label_dir,val_examples)


# In[3]:


def create_record(output_filename,image_dir,label_dir,examples):
    '''Generate pictures into tfrecord
     parameter:
       output_filename: output address
       image_dir: image address
       label_dir: label address
       examples: the index name of the image
      '''
    writer=tf.python_io.TFRecordWriter(output_filename)
    for idx,example in enumerate(examples):
        if idx % 500 ==0:
            tf.logging.info('On image %d of %d',idx,len(examples))
        image_path=os.path.join(image_dir,example+'.jpg')
        label_path=os.path.join(label_dir,example+'.png')
        
        if not os.path.exists(image_path):
            tf.logging.warning('invalid image: ',image_path)
            continue
        elif not os.path.exists(label_path):
            tf.logging.warning('invalid label ',label_path)
            continue
        try:
            
            tf_example=dict_to_tf_example(image_path,label_path)
           
            writer.write(tf_example.SerializeToString())
        except ValueError:
            tf.logging.warning('Invalid example: %s, ignore',example)
    writer.close()


# In[4]:


def dict_to_tf_example(image_path,label_path):
    '''format to tfrecord
     parameter:
       image_path: Enter the image address
       label_path: output label address
      '''
    with tf.gfile.GFile(image_path,'rb') as f:
        encoder_jpg=f.read()
    encoder_jpg_io=io.BytesIO(encoder_jpg)
    image=Image.open(encoder_jpg_io)
  
    if image.format !='JPEG':
        tf.logging.info('input image format error')
        raise ValueError('input image format error')
    
    with tf.gfile.GFile(label_path,'rb') as f:
        encoder_label=f.read()
    encoder_label_io=io.BytesIO(encoder_label)
    label=Image.open(encoder_label_io)
    
    if label.format !='PNG':
        tf.logging.info('label image format error')
        raise ValueError('label image format error')
    
    if image.size!=label.size:
        tf.logging.info('Unmatched input and output')
        raise ValueError('Unmatched input and output')
   
    example=tf.train.Example(features=tf.train.Features(feature={
        'image':dataset_util.bytes_feature(encoder_jpg),
        'label':dataset_util.bytes_feature(encoder_label)}))
    return example
    


# In[5]:


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

