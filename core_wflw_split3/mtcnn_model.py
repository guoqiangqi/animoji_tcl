import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import random
import cv2
import math
num_keep_radio = 0.7

def wing_loss(landmark_pred, landmark_target,label,animoji_target=None,w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks*2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    # with tf.name_scope('wing_loss'):
        # x = landmark_pred - landmark_target
        # c = w * (1.0 - math.log(1.0 + w/epsilon))
        # absolute_x = tf.abs(x)
        # losses = tf.where(
            # tf.greater(w, absolute_x),
            # w * tf.log(1.0 + absolute_x/epsilon),
            # absolute_x - c
        # )
        # loss = tf.reduce_mean(tf.reduce_sum(losses, axis=1), axis=0)
        # return loss
    
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-3),ones,zeros)
    x = landmark_pred - landmark_target
    c = w * (1.0 - math.log(1.0 + w/epsilon))
    absolute_x = tf.abs(x)
    losses = tf.where(
        tf.greater(w, absolute_x),
        w * tf.log(1.0 + absolute_x/epsilon),
        absolute_x - c
        )
    losses = tf.reduce_sum(losses,axis=1)   
    
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    losses = losses*valid_inds
    _, k_index = tf.nn.top_k(losses, k=keep_num)    
    losses = tf.gather(losses, k_index)
    if animoji_target is not None:  
        landmark_num=18.0
        left =tf.concat([animoji_target[:,120:136],animoji_target[:,192:194],animoji_target[:,66:84]],1)
        right=tf.concat([animoji_target[:,136:152],animoji_target[:,194:196],animoji_target[:,84:102]],1)             
        
        standard_loss=tf.reduce_sum(tf.square(left-right),axis=1)**0.5/landmark_num
        standard_loss=standard_loss*valid_inds
        standard_loss=tf.gather(standard_loss,k_index)
        losses=tf.div(losses,standard_loss)    
    else:
        pass
    return tf.reduce_mean(losses)

def prelu(inputs):
    alphas = tf.get_variable('alphas', shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
    
def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def animoji_ohem(landmark_pred,landmark_target,label,animoji_target=None):
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-3),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    if animoji_target is not None:  
        landmark_num=18.0
        left =tf.concat([animoji_target[:,120:136],animoji_target[:,192:194],animoji_target[:,66:84]],1)
        right=tf.concat([animoji_target[:,136:152],animoji_target[:,194:196],animoji_target[:,84:102]],1)             
        
        standard_loss=tf.reduce_sum(tf.square(left-right),axis=1)**0.5/landmark_num
        standard_loss=standard_loss*valid_inds
        standard_loss=tf.gather(standard_loss,k_index)
        square_error=tf.div(square_error,standard_loss)    
    else:
        pass
    return tf.reduce_mean(square_error)   
    
def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    
    cond = tf.where(tf.greater(label_picked,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_picked,picked)
    pred_picked = tf.gather(pred_picked,picked)
    recall_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op,recall_op

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        print('P_Net network shape')
        print(inputs.get_shape())
        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        print(net.get_shape())
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        
        print(conv4_1.get_shape())
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        print(bbox_pred.get_shape())

        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        print(landmark_pred.get_shape())

        if training:
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name='landmark_pred')
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            accuracy, recall = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test
        
def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print('R_Net network shape')
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope='conv1')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool1', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope='conv2')
        print(net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope='conv3')
        print(net.get_shape())
        net = tf.transpose(net, perm=[0,3,1,2]) 
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope='fc1', activation_fn=prelu)
        print(fc1.get_shape())

        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())

        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        print(bbox_pred.get_shape())

        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
        print(landmark_pred.get_shape())

        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy, recall = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            # return cls_prob,bbox_pred,landmark_pred,None
            return cls_prob,bbox_pred,None
    
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print('O_Net network shape')
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope='conv1')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool1', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv2')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv3')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope='conv4')
        print(net.get_shape())
        net = tf.transpose(net, perm=[0,3,1,2]) 
        print(net.get_shape())        
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope='fc1', activation_fn=prelu)
        print(fc1.get_shape())

        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())

        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        print(bbox_pred.get_shape())

        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
        print(landmark_pred.get_shape())        

        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy, recall = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            return cls_prob,bbox_pred,landmark_pred,None

# def L_O_Net(inputs,label=None,bbox_target=None,landmark_target=None,animoji_target=None,training=True):
def L_O_Net(inputs,label=None,bbox_target=None,animoji_target=None,training=True):
    # batch_norm_params = {
        # 'decay': 0.995,
        # 'epsilon': 0.001,
        # 'updates_collections': None,
        # 'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    # }
    # with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        # activation_fn = prelu,
                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        # biases_initializer=tf.zeros_initializer(),
                        # weights_regularizer=slim.l2_regularizer(0.0005),
                        # normalizer_fn=slim.batch_norm,    
                        # normalizer_params=batch_norm_params
                        # ):   
    
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print('L_O_Net network shape')
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=2, scope='conv1')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool1', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv2')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope='conv3')
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3', padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope='conv4')
        print(net.get_shape())
       
        net = tf.transpose(net, perm=[0,3,1,2]) 
        print(net.get_shape())        
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope='fc1', activation_fn=prelu)
        print(fc1.get_shape())

        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope='cls_fc',activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())

        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope='bbox_fc',activation_fn=None)
        print(bbox_pred.get_shape())

        # landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope='landmark_fc',activation_fn=None)
        # print(landmark_pred.get_shape())        

        # animoji_pred = slim.fully_connected(fc1,num_outputs=196,scope='animoji_fc',activation_fn=None)
        # print(animoji_pred.get_shape())
        # eye_pred = slim.fully_connected(fc1,num_outputs=36,scope='eye_fc',activation_fn=None)
        # print(eye_pred.get_shape())
        eye_eyebrow_pred = slim.fully_connected(fc1,num_outputs=72,scope='eyebrow_fc',activation_fn=None)
        print(eye_eyebrow_pred.get_shape())
        mouth_pred = slim.fully_connected(fc1,num_outputs=40,scope='mouth_fc',activation_fn=None)
        print(mouth_pred.get_shape())
        multi_pred = slim.fully_connected(fc1,num_outputs=84,scope='multi_fc',activation_fn=None)
        print(multi_pred.get_shape())
        
        if training:
             #get target label 
            eye_target=tf.concat([animoji_target[:,120:152],animoji_target[:,192:196]],1)            
            eyebrow_target=animoji_target[:,66:102]
            eye_eyebrow_target=tf.concat([eye_target,eyebrow_target],1)            
            mouth_target=animoji_target[:,152:192]
            multi_target=tf.concat([animoji_target[:,0:66],animoji_target[:,102:120]],1)
            
            # left =tf.concat([animoji_target[:,120:136],animoji_target[:,192:194],animoji_target[:,66:84]],1)
            # right=tf.concat([animoji_target[:,136:152],animoji_target[:,194:196],animoji_target[:,84:102]],1)             
            # standard_loss=tf.reduce_sum(tf.square(left-right),axis=1)#**0.5/landmark_num
            
            cls_loss = cls_ohem(cls_prob,label)
            accuracy, recall = cal_accuracy(cls_prob,label)
            # landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            # animoji_loss = animoji_ohem(animoji_pred, animoji_target,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            eye_eyebrow_loss = animoji_ohem(eye_eyebrow_pred, eye_eyebrow_target,label,animoji_target=animoji_target)
            mouth_loss = animoji_ohem(mouth_pred, mouth_target,label) 
            multi_loss = animoji_ohem(multi_pred, multi_target,label)
            # bbox_loss = wing_loss(bbox_pred,bbox_target,label)
            # eye_eyebrow_loss = wing_loss(eye_eyebrow_pred, eye_eyebrow_target,label,animoji_target=animoji_target)
            # mouth_loss = wing_loss(mouth_pred, mouth_target,label) 
            # multi_loss = wing_loss(multi_pred, multi_target,label)

            print(tf.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            # return cls_loss,bbox_loss,landmark_loss,animoji_loss,L2_loss,accuracy, recall
            return cls_loss,bbox_loss,eye_eyebrow_loss,mouth_loss,multi_loss,L2_loss,accuracy, recall
        else:
            # return cls_prob,bbox_pred,landmark_pred,animoji_pred
            return cls_prob,bbox_pred,eye_eyebrow_pred,mouth_pred,multi_pred
            
def train_model(base_lr, loss, data_num, batch_size, lr_epoch):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / batch_size) for epoch in lr_epoch]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(lr_epoch) + 1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    # optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    optimizer = tf.train.AdamOptimizer(lr_op)
    train_op = optimizer.minimize(loss, global_step)

    return train_op, lr_op

# def random_flip_images(image_batch,label_batch,landmark_batch):
def random_flip_images(image_batch,label_batch):
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
    
        # for i in fliplandmarkindexes:
            # landmark_ = landmark_batch[i].reshape((-1,2))
            # landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            # landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            # landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            # landmark_batch[i] = landmark_.ravel()
        
    return image_batch      
        
def read_single_tfrecord(tfrecord_file, batch_size, image_size, gray=False):
    # filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # image_features = tf.parse_single_example(
        # serialized_example,
        # features={
            # 'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            # 'image/label': tf.FixedLenFeature([], tf.int64),
            # 'image/roi': tf.FixedLenFeature([4], tf.float32),
            # 'image/landmark': tf.FixedLenFeature([10],tf.float32),
            # 'image/animoji': tf.FixedLenFeature([140],tf.float32)
        # }
    # )

    # image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    # image = tf.reshape(image, [image_size, image_size, 3])
    # if gray:
        # image = tf.image.rgb_to_grayscale(image)
    # image = (tf.cast(image, tf.float32)-127.5) / 128
    
    # label = tf.cast(image_features['image/label'], tf.float32)
    # roi = tf.cast(image_features['image/roi'],tf.float32)
    # landmark = tf.cast(image_features['image/landmark'],tf.float32)
    # animoji = tf.cast(image_features['image/animoji'],tf.float32)
    # image, label,roi,landmark,animoji = tf.train.batch(
        # [image, label,roi,landmark,animoji],
        # batch_size=batch_size,
        # num_threads=2,
        # capacity=1 * batch_size
    # )
    # label = tf.reshape(label, [batch_size])
    # roi = tf.reshape(roi,[batch_size,4])
    # landmark = tf.reshape(landmark,[batch_size,10])
    # animoji = tf.reshape(animoji,[batch_size,140])
    # return image, label, roi,landmark,animoji
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10],tf.float32),
            'image/animoji': tf.FixedLenFeature([196],tf.float32)
        }
    )

    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    if gray:
        image = tf.image.rgb_to_grayscale(image)
    image = (tf.cast(image, tf.float32)-127.5) / 128
    
    label = tf.cast(image_features['image/label'], tf.float32)
    roi = tf.cast(image_features['image/roi'],tf.float32)
    # landmark = tf.cast(image_features['image/landmark'],tf.float32)
    animoji = tf.cast(image_features['image/animoji'],tf.float32)
    image, label,roi,animoji = tf.train.batch(
        [image, label,roi,animoji],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size])
    roi = tf.reshape(roi,[batch_size,4])
    # landmark = tf.reshape(landmark,[batch_size,10])
    animoji = tf.reshape(animoji,[batch_size,196])
    return image, label, roi,animoji

def read_multi_tfrecords(tfrecord_files, batch_sizes, net, gray=False):
    # pos_dir,part_dir,neg_dir,landmark_dir,animoji_dir = tfrecord_files
    # pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size,animoji_batch_size = batch_sizes
    # pos_image,pos_label,pos_roi,pos_landmark,pos_animoji = read_single_tfrecord(pos_dir, pos_batch_size, net, gray)
    # print(pos_image.get_shape())
    # part_image,part_label,part_roi,part_landmark,part_animoji = read_single_tfrecord(part_dir, part_batch_size, net, gray)
    # print(part_image.get_shape())
    # neg_image,neg_label,neg_roi,neg_landmark,net_animoji = read_single_tfrecord(neg_dir, neg_batch_size, net, gray)
    # print(neg_image.get_shape())
    # landmark_image,landmark_label,landmark_roi,landmark_landmark,landmark_animoji = read_single_tfrecord(landmark_dir, landmark_batch_size, net, gray)
    # print(landmark_image.get_shape())
    
    # if animoji_dir is None:
        # images = tf.concat([pos_image,part_image,neg_image,landmark_image], 0, name='concat/image')
        # print(images.get_shape())
        # labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name='concat/label')
        # print(labels.get_shape())
        # rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name='concat/roi')
        # print(rois.get_shape())
        # landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name='concat/landmark')
        # print(landmarks.get_shape())
        # return images,labels,rois,landmarks,None
    # else:        
        # animoji_image,animoji_label,animoji_roi,animoji_landmark,animoji_animoji = read_single_tfrecord(animoji_dir, animoji_batch_size, net, gray)
        # print(animoji_image.get_shape())
        
        # images = tf.concat([pos_image,part_image,neg_image,landmark_image,animoji_image], 0, name='concat/image')
        # print(images.get_shape())
        # labels = tf.concat([pos_label,part_label,neg_label,landmark_label,animoji_label],0,name='concat/label')
        # print(labels.get_shape())
        # rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi,animoji_roi],0,name='concat/roi')
        # print(rois.get_shape())
        # landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark,animoji_landmark],0,name='concat/landmark')
        # print(landmarks.get_shape())
        # animojis = tf.concat([pos_animoji,part_animoji,net_animoji,landmark_animoji,animoji_animoji],0,name='concat/animoji')
        # print(animojis.get_shape())
        # # return images,labels,rois,landmarks,animojis
        # return images,labels,rois,animojis
    pos_dir,part_dir,neg_dir,animoji_dir = tfrecord_files
    pos_batch_size,part_batch_size,neg_batch_size,animoji_batch_size = batch_sizes
    pos_image,pos_label,pos_roi,pos_animoji = read_single_tfrecord(pos_dir, pos_batch_size, net, gray)
    print(pos_image.get_shape())
    part_image,part_label,part_roi,part_animoji = read_single_tfrecord(part_dir, part_batch_size, net, gray)
    print(part_image.get_shape())
    neg_image,neg_label,neg_roi,net_animoji = read_single_tfrecord(neg_dir, neg_batch_size, net, gray)
    print(neg_image.get_shape())
    # landmark_image,landmark_label,landmark_roi,landmark_animoji = read_single_tfrecord(landmark_dir, landmark_batch_size, net, gray)
    # print(landmark_image.get_shape())
    
    if animoji_dir is None:
        images = tf.concat([pos_image,part_image,neg_image], 0, name='concat/image')
        print(images.get_shape())
        labels = tf.concat([pos_label,part_label,neg_label],0,name='concat/label')
        print(labels.get_shape())
        rois = tf.concat([pos_roi,part_roi,neg_roi],0,name='concat/roi')
        print(rois.get_shape())
        # landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name='concat/landmark')
        # print(landmarks.get_shape())
        return images,labels,rois,None
    else:        
        animoji_image,animoji_label,animoji_roi,animoji_animoji = read_single_tfrecord(animoji_dir, animoji_batch_size, net, gray)
        print(animoji_image.get_shape())
        
        images = tf.concat([pos_image,part_image,neg_image,animoji_image], 0, name='concat/image')
        print(images.get_shape())
        labels = tf.concat([pos_label,part_label,neg_label,animoji_label],0,name='concat/label')
        print(labels.get_shape())
        rois = tf.concat([pos_roi,part_roi,neg_roi,animoji_roi],0,name='concat/roi')
        print(rois.get_shape())
        # landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark,animoji_landmark],0,name='concat/landmark')
        # print(landmarks.get_shape())
        animojis = tf.concat([pos_animoji,part_animoji,net_animoji,animoji_animoji],0,name='concat/animoji')
        print(animojis.get_shape())
        # return images,labels,rois,landmarks,animojis
        return images,labels,rois,animojis
   
        