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

pre_model_path = None
root_dir = os.path.dirname(os.path.realpath(__file__)) 
base_dir = os.path.join(root_dir, '../data_landmark98/training_data/wflw_98_L94/imglists/LONet')
model_path = os.path.join(root_dir, '../models/LONet_wflw_98_L94_finetune/LONet')
logs_dir = os.path.join(root_dir, '../models/logs/LONet_wflw_98_L94_finetune')
# pre_model_path = os.path.join(root_dir, '../models/LONet_wflw_finetune_1/LONet-1600')
pre_model_path = os.path.join(root_dir, '../models/LONet_wflw_98_L94/LONet-11')
print('root_dir: ', root_dir)
print('base_dir: ', base_dir)
print('model_path: ', model_path)
print('logs_dir: ', logs_dir)
        
prefix = model_path
end_epoch = 3000
display = 500
base_lr = 0.001
BATCH_SIZE = 357
image_size = 94
# image_size = 96
ratio_cls_loss = 1.0
ratio_bbox_loss = 1.0
# ratio_landmark_loss = 1.0
# ratio_animoji_loss = 1.0
ratio_eye_loss = 1.5
ratio_eyebrow_loss = 1.5
ratio_mouth_loss = 1.5
ratio_multi_loss = 1.0
# LR_EPOCH = [6,14,20,30,50,80,120]
# LR_EPOCH = [10,40,100,300,600,1000,2000,5000,10000] 
LR_EPOCH = [6,7,8,10,120,170,230]
net = 'LONet'
label_file = os.path.join(base_dir,'train_{}.txt'.format(net))
print(label_file) 
f = open(label_file, 'r')
num = len(f.readlines())
print('Total datasets is: ', num)
print(prefix)
gray = False
if gray:
    image_channel = 1
else:
    image_channel = 3
print('iamge channel: ', image_channel)
pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
# landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
animoji_dir = os.path.join(base_dir,'animoji_landmark.tfrecord_shuffle')
# dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir,animoji_dir]
# pos_ratio = 1.0/9;part_ratio = 1.0/9;landmark_ratio=2.0/9;neg_ratio=3.0/9;animoji_ratio=2.0/9
dataset_dirs = [pos_dir,part_dir,neg_dir,animoji_dir]
pos_ratio = 1.0/7;part_ratio = 1.0/7;neg_ratio=3.0/7;animoji_ratio=2.0/7
pos_batch_size = int(np.ceil(BATCH_SIZE*pos_ratio))
assert pos_batch_size != 0,'Batch Size Error '
part_batch_size = int(np.ceil(BATCH_SIZE*part_ratio))
assert part_batch_size != 0,'Batch Size Error '        
neg_batch_size = int(np.ceil(BATCH_SIZE*neg_ratio))
assert neg_batch_size != 0,'Batch Size Error '
# landmark_batch_size = int(np.ceil(BATCH_SIZE*landmark_ratio))
# assert landmark_batch_size != 0,'Batch Size Error '
animoji_batch_size = int(np.ceil(BATCH_SIZE*animoji_ratio))
assert animoji_batch_size != 0,'Batch Size Error '
# batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size,animoji_batch_size]
# image_batch, label_batch, bbox_batch,landmark_batch,animoji_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, image_size, gray)
batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,animoji_batch_size]
image_batch, label_batch, bbox_batch,animoji_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, image_size, gray)
input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size, image_size, image_channel], name='input_image')
label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
bbox_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name='bbox_target')
# landmark_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,10],name='landmark_target')
animoji_target = tf.placeholder(tf.float32,shape=[BATCH_SIZE,196],name='animoji_target')
# cls_loss_op,bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,accuracy_op,recall_op = L_O_Net(input_image, label, bbox_target,landmark_target,animoji_target,training=True)
# loss = ratio_cls_loss*cls_loss_op + ratio_bbox_loss*bbox_loss_op + ratio_landmark_loss*landmark_loss_op + animoji_loss_op*ratio_animoji_loss + L2_loss_op
cls_loss_op,bbox_loss_op,eye_loss_op,eyebrow_loss_op,mouth_loss_op,multi_loss_op,L2_loss_op,accuracy_op, recall_op = L_O_Net(input_image, label, bbox_target,animoji_target,training=True)
loss = ratio_cls_loss*cls_loss_op + ratio_bbox_loss*bbox_loss_op \
     + ratio_eye_loss*eye_loss_op +ratio_eyebrow_loss*eyebrow_loss_op +ratio_mouth_loss*mouth_loss_op+ratio_multi_loss*multi_loss_op \
     + L2_loss_op
train_op, lr_op = train_model(base_lr, loss, num, BATCH_SIZE, LR_EPOCH)

init = tf.global_variables_initializer()
sess = tf.Session(config=config)

saver = tf.train.Saver(max_to_keep=0)
sess.run(init)

tf.summary.scalar('cls_loss',cls_loss_op)
tf.summary.scalar('bbox_loss',bbox_loss_op)
# tf.summary.scalar('landmark_loss',landmark_loss_op)
# tf.summary.scalar('animoji_loss',animoji_loss_op)
tf.summary.scalar('eye_loss',eye_loss_op)
tf.summary.scalar('eyebrow_loss',eyebrow_loss_op)
tf.summary.scalar('mouth_loss',mouth_loss_op)
tf.summary.scalar('multi_loss',multi_loss_op)
tf.summary.scalar('cls_accuracy',accuracy_op)
tf.summary.scalar('cls_recall',recall_op)
summary_op = tf.summary.merge_all()

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
writer = tf.summary.FileWriter(logs_dir,sess.graph)
#begin 
coord = tf.train.Coordinator()
#begin enqueue thread
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
i = 0
#total steps
MAX_STEP = int(num / BATCH_SIZE + 1) * end_epoch
epoch = 0
sess.graph.finalize()    
try:
    if pre_model_path is not None:
        print('pre_training model: ', pre_model_path)
        saver.restore(sess, pre_model_path)
    for step in range(MAX_STEP):
        i = i + 1
        if coord.should_stop():
            break
        # image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array,animoji_batch_array= \
                        # sess.run([image_batch, label_batch, bbox_batch,landmark_batch,animoji_batch])
        image_batch_array, label_batch_array, bbox_batch_array,animoji_batch_array= \
                        sess.run([image_batch, label_batch, bbox_batch,animoji_batch])
                        
        # for img_idx, img in enumerate(image_batch_array):
            # img = img *128 + 127.5
            # img = img.astype(np.uint8)
            # cv2.imwrite('outimg/{}.jpg'.format(img_idx), img)
        # exit()
        #random flip
        # image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
        image_batch_array = random_flip_images(image_batch_array,label_batch_array,)
        
        # _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array,\
                                            # bbox_target: bbox_batch_array,landmark_target:landmark_batch_array, \
                                            # animoji_target: animoji_batch_array})
                                            
        _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array,\
                                            bbox_target: bbox_batch_array, \
                                            animoji_target: animoji_batch_array})
        
        # if (step+1) % display == 0:
            # cls_loss, bbox_loss,landmark_loss,animoji_loss,L2_loss,lr,acc,recall = \
                # sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                # feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                # bbox_batch_array, landmark_target: landmark_batch_array, \
                # animoji_target: animoji_batch_array})
                
        if (step+1) % display == 0:
            cls_loss, bbox_loss,eye_loss,eyebrow_loss,mouth_loss,multi_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,eye_loss_op,eyebrow_loss_op,mouth_loss_op,multi_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array,animoji_target: animoji_batch_array})
                
            # print('{}: Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format(
                # datetime.now(), step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, animoji_loss, L2_loss, lr))              
            print('{}: Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, eye loss: {:.4},eyebrow loss: {:.4},mouth loss: {:.4},multi loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format(
                datetime.now(), step+1, acc, recall, cls_loss, bbox_loss,  eye_loss,eyebrow_loss,mouth_loss,multi_loss, L2_loss, lr))

        # if i * BATCH_SIZE > num:
            # epoch = epoch + 1
            # i = 0
            # cls_loss, bbox_loss,landmark_loss,animoji_loss,L2_loss,lr,acc,recall = \
                # sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,animoji_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                # feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                # bbox_batch_array, landmark_target: landmark_batch_array, animoji_target: animoji_batch_array})
            # print('{}: {}, Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4}, landmark loss: {:.4}, animoji loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format( \
                # 'Save model epoch', epoch, step+1, acc, recall, cls_loss, bbox_loss, landmark_loss, animoji_loss, L2_loss, lr))
            # saver.save(sess, prefix, global_step=epoch)
            # # saver.save(sess, prefix, global_step=step)
        # writer.add_summary(summary,global_step=step)
        if i * BATCH_SIZE > num:
            epoch = epoch + 1
            i = 0
            cls_loss, bbox_loss,eye_loss,eyebrow_loss,mouth_loss,multi_loss,L2_loss,lr,acc,recall = \
                sess.run([cls_loss_op, bbox_loss_op,eye_loss_op,eyebrow_loss_op,mouth_loss_op,multi_loss_op,L2_loss_op,lr_op,accuracy_op,recall_op],
                feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: \
                bbox_batch_array,animoji_target: animoji_batch_array})
            print('{}: {}, Step: {}, acc: {:.3}, recall: {:.3}, cls loss: {:.4}, bbox loss: {:.4},  eye loss: {:.4},eyebrow loss: {:.4},mouth loss: {:.4},multi loss: {:.4}, L2 loss: {:.4}, lr:{:.4} '.format( \
                'Save model epoch', epoch, step+1, acc, recall, cls_loss, bbox_loss, eye_loss,eyebrow_loss,mouth_loss,multi_loss, L2_loss, lr))
            saver.save(sess, prefix, global_step=epoch)
            # saver.save(sess, prefix, global_step=step)
        writer.add_summary(summary,global_step=step)
except tf.errors.OutOfRangeError:
    print('finish.')
finally:
    coord.request_stop()
    writer.close()
coord.join(threads)
sess.close()
print('finish.')
