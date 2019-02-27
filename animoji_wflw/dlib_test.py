import cv2
import dlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../core'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data'))
from data_utils import IoU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from detection_py import detect_face


root_dir = os.path.dirname(os.path.realpath(__file__))
model_path = [os.path.join(root_dir, '../models/PNet_1/PNet-60'), 
              os.path.join(root_dir, '../models/RNet_1/RNet-42'),
              os.path.join(root_dir, '../models/LONet_1/LONet-14')]             
              
# face_model_path = [os.path.join(root_dir, 'models/face/-1716'),
#                    os.path.join(root_dir, 'models/eye/-682'),
#                    os.path.join(root_dir, 'models/mouse/-714'),
#                    os.path.join(root_dir, 'models/nose/-1058')]

facenet = detect_face(model_path, True)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
 
imagepath = os.path.join(root_dir, '../image_test/001.jpg')
image = cv2.imread(imagepath)
h,w,_ = image.shape
if min(h,w) > 1000:
    image = cv2.resize(image, (w//2, h//2))

boxes, landmarks, animojis = facenet.predict(image.copy())
print('number boxes: ', len(boxes))
for box in boxes:
    x1,y1,x2,y2,_ = box
    face = dlib.rectangle(int(x1+0.5), int(y1+0.5), int(x2+0.5), int(y2+0.5))
    shape = predictor(image, face)
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(image, pt_pos, 2, (0, 255, 0), 1)

cv2.imshow('dlib_image.jpg', image)
c =cv2.waitKey(0)
if c &0xff == ord(c):
    cv2.destroyAllWindows()