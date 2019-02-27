#coding:utf-8
import numpy as np
import os
import cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
from datetime import datetime
from mtcnn_model import O_Net,L_O_Net,read_multi_tfrecords,train_model,random_flip_images

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

root_dir = os.path.dirname(os.path.realpath(__file__)) 
base_dir = os.path.join(root_dir, '../data_landmark70/training_data/wflw_70_L94/imglists/LONet')
# eval_model_path = os.path.join(root_dir, '../models/LONet_wflw_98_L94/LONet-6')
        
BATCH_SIZE = 357
image_size = 94

net = 'LONet'
label_file = os.path.join(base_dir,'train_{}.txt'.format(net))
print(label_file) 
f = open(label_file, 'r')
num = len(f.readlines())
print('Total datasets is: ', num)

gray = False
if gray:
    image_channel = 1
else:
    image_channel = 3
print('iamge channel: ', image_channel)

pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
animoji_dir = os.path.join(base_dir,'animoji_landmark.tfrecord_shuffle')

dataset_dirs = [pos_dir,part_dir,neg_dir,animoji_dir]
pos_ratio = 1.0/7;part_ratio = 1.0/7;neg_ratio=3.0/7;animoji_ratio=2.0/7

pos_batch_size = int(np.ceil(BATCH_SIZE*pos_ratio))
assert pos_batch_size != 0,'Batch Size Error '
part_batch_size = int(np.ceil(BATCH_SIZE*part_ratio))
assert part_batch_size != 0,'Batch Size Error '        
neg_batch_size = int(np.ceil(BATCH_SIZE*neg_ratio))
assert neg_batch_size != 0,'Batch Size Error '
animoji_batch_size = int(np.ceil(BATCH_SIZE*animoji_ratio))
assert animoji_batch_size != 0,'Batch Size Error '

batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,animoji_batch_size]
image_batch, label_batch, bbox_batch,animoji_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, image_size, gray)
input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, image_channel], name='input_image')
label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
bbox_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name='bbox_target')
animoji_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,140],name='animoji_target')

cls_loss_op,bbox_loss_op,animoji_loss_op,L2_loss_op,accuracy_op, recall_op = L_O_Net(input_image, label, bbox_target,animoji_target,training=True)

init = tf.global_variables_initializer()
sess = tf.Session(config=config)

saver = tf.train.Saver(max_to_keep=0)
sess.run(init)

#begin 
coord = tf.train.Coordinator()
#begin enqueue thread
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#total steps
MAX_STEP = int(num / BATCH_SIZE + 1)
# epoch = 0
sess.graph.finalize()

best_cls_model_number=0
best_bbox_model_number=0
best_animoji_model_number=0
min_ave_cls_loss=100
min_ave_bbox_loss=100
min_ave_animoji_loss=100

for i in range(1,62):
    eval_model_path=os.path.join(root_dir, '../models/LONet_wflw_98_L94_finetune/LONet-')
    eval_model_path=eval_model_path+str(i)
    try:
        if eval_model_path is not None:
            print('eval model: ', eval_model_path)
            saver.restore(sess, eval_model_path)
        
        total_animoji_loss=0
        total_cls_loss=0
        total_bbox_loss=0
        
        MAX_STEP=100
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            # image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array,animoji_batch_array= \
                            # sess.run([image_batch, label_batch, bbox_batch,landmark_batch,animoji_batch])
            image_batch_array, label_batch_array, bbox_batch_array,animoji_batch_array= \
                            sess.run([image_batch, label_batch, bbox_batch,animoji_batch])

          
            image_batch_array = random_flip_images(image_batch_array,label_batch_array,)
                            
            cls_loss,bbox_loss,animoji_loss,L2_loss,acc,recall = \
                sess.run([cls_loss_op,bbox_loss_op,animoji_loss_op,L2_loss_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array,animoji_target: animoji_batch_array})
                    
                # print('{}: Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format(
                    # datetime.now(), step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, animoji_loss, L2_loss, lr))              
            # print('{}: Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4} '.format(
                # datetime.now(), step+1, acc, recall,cls_loss, bbox_loss,animoji_loss, L2_loss))
            
            total_animoji_loss=total_animoji_loss+animoji_loss
            total_cls_loss=total_cls_loss+cls_loss
            total_bbox_loss=total_bbox_loss+bbox_loss
        ave_animoji_loss=total_animoji_loss/MAX_STEP
        ave_cls_loss=total_cls_loss/MAX_STEP
        ave_bbox_loss=total_bbox_loss/MAX_STEP
        print('eval_model_number: {},ave_cls_loss: {},ave_bbox_loss: {},ave_animoji_loss: {}'\
              .format(i,ave_cls_loss,ave_bbox_loss,ave_animoji_loss))
        
        if(ave_animoji_loss<min_ave_animoji_loss):
            min_ave_animoji_loss=ave_animoji_loss
            best_animoji_model_number=i
        
         
    except tf.errors.OutOfRangeError:
        print('finish.')
    # finally:
        # coord.request_stop()
coord.request_stop()       
coord.join(threads)
sess.close()
print('best_animoji_model_number: {},min_animoji_loss:{}'.format(best_animoji_model_number,min_ave_animoji_loss))
print('finish.')
