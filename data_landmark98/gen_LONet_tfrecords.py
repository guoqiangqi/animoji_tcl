import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import random
import sys
import time
import shutil
import tensorflow as tf
from tfrecord_utils import add_to_tfrecord, get_dataset
from gen_48net_data import gen_ONet_bbox_data
from generateLandmark import GenLandmarkData
    
def run(imagelist, tf_filename, shuffling=False):

    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    dataset = get_dataset(imagelist)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 5000 == 0:
                print('\r>> Converting image {}/{}'.format(i + 1, len(dataset)))
            filename = image_example['filename']
            add_to_tfrecord(filename, image_example, tfrecord_writer)
    print('\nFinished converting the MTCNN dataset!')

    
def gen_ONet_tfrecords(
    bbox_anno_file,
    bbox_im_dir,
    save_dir,
    landmark_anno_file,
    animoji_txt,
    tfrecords_output_dir,
    model_path,
    debug=False,
    image_size=48):
    
    #size = 48
    size =image_size
    net = 'LONet'
    
    # pos_list_file, neg_list_file, part_list_file = files
    files = gen_ONet_bbox_data(bbox_anno_file, bbox_im_dir, save_dir, model_path, debug=debug,image_size=size)
    _,_,landmark_list_file = GenLandmarkData(landmark_anno_file, net, size, save_dir, \
                                             argument=True,debug=debug)
    _,_,animoji_list_file = GenLandmarkData(animoji_txt, 'animoji_'+net,size,save_dir, \
                                             argument=True,debug=debug)
    
    with open(files[0], 'r') as f:
        pos = f.readlines()

    with open(files[1], 'r') as f:
        neg = f.readlines()

    with open(files[2], 'r') as f:
        part = f.readlines()

    with open(landmark_list_file, 'r') as f:
        landmark = f.readlines()
    
    with open(animoji_list_file, 'r') as f:
        animoji = f.readlines()
    #write all data
    imageLists = [pos, neg, part, landmark, animoji]
    if not os.path.exists(tfrecords_output_dir):
        os.makedirs(tfrecords_output_dir)
        
    with open(os.path.join(tfrecords_output_dir, "train_{}.txt".format(net)), "w") as f:
        print(len(neg))
        print(len(pos))
        print(len(part))
        print(len(landmark))
        print(len(animoji))
        for i in np.arange(len(pos)):
            f.write(pos[i])
        for i in np.arange(len(neg)):
            f.write(neg[i])
        for i in np.arange(len(part)):
            f.write(part[i])
        for i in np.arange(len(landmark)):
            f.write(landmark[i])
        for i in np.arange(len(animoji)):
            f.write(animoji[i])
            
    tf_filenames = [
        os.path.join(tfrecords_output_dir,'pos_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'part_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'neg_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'landmark_landmark.tfrecord'),
        os.path.join(tfrecords_output_dir,'animoji_landmark.tfrecord'),
    ]    
    
    for imgs, files in zip(imageLists, tf_filenames):
        run(imgs, files, shuffling=True)
         
if __name__ == '__main__': 
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    anno_file = os.path.join(root_dir, 'training_data/wider_face/wider_face_train_bbx_gt.txt')
    im_dir = os.path.join(root_dir, 'training_data/wider_face/images')
    landmark_txt = os.path.join(root_dir, 'training_data/landmarks_5/trainImageList.txt')
    animoji_txt = os.path.join(root_dir, 'training_data/landmarks_70/ImageList_98points.txt')
    # train_root = os.path.join(root_dir, '')
    model_path = [os.path.join(root_dir, '../models/PNet_1/PNet-60'),
                  os.path.join(root_dir, '../models/RNet_1/RNet-42')]
  
    save_dir = os.path.join(root_dir, 'training_data/L224_wflw')
    tfrecords_output_dir = os.path.join(root_dir, 'training_data/L224_wflw/imglists/LONet')              
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists(tfrecords_output_dir):
        shutil.rmtree(tfrecords_output_dir)
		
    image_size=224
    gen_ONet_tfrecords(anno_file,      \
                       im_dir,          \
                       save_dir,         \
                       landmark_txt,      \
                       animoji_txt,        \
                       tfrecords_output_dir,\
                       model_path,           \
                       debug=False,
                       image_size=image_size)				   