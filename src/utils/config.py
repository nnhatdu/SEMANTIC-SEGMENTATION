num_classes=21
data_dir='./data/VOCdevkit/VOC2012'
tfrecord_path='./data/tfrecord/'
train_data_list='./data/train.txt'
val_data_list='./data/val.txt'
image_data_dir='JPEGImages'
label_data_dir='SegmentationClassAug'

model_dir='./model'
clean_model_dir='store_false'
train_epochs=2
epochs_per_eval=1

tensorboard_images_max_outputs=6

batch_size=4
learning_rate_policy='poly'
max_iter=30000

base_architecture='resnet_v2_101'
pre_trained_model='./resnet_v2_101/resnet_v2_101.ckpt'
output_stride=16
freeze_batch_norm='store_true'
initial_learning_rate=7e-3
end_learning_rate=1e-6
initial_global_step=0
weight_decay=2e-4

debug=None

pictue='./picture/'
output='./output/'
test_mode='2'