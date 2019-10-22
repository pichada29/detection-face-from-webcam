import cv2
import numpy as np
import os
from skimage import feature
from skimage import transform
from sklearn.metrics.pairwise import cosine_similarity
import glob
import fnmatch
import time

import keras
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')

x_data = []
y_data = []
y_full_name = []

thresh = 0.5
simm = []
simm_label = []
simm_label_fullpath = []

orientations = 8
pixels_per_cell = (8,8)
cells_per_block = (1,1)
cnt = 1
n_rows, n_cols = [224,224]

path = 'your destination'
matches = []

# for loop ในการดึงภาพออกมาจาก dir
for root, dirnames, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, '*.[jpg]*[gG]'):
        matches.append(os.path.join(root,filename))

# function ในการ read image ด้วยการใช้ cv2
def return_image(file_loc):
  
  image = cv2.imread(file_loc, 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  color_img = np.array(image)
  gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
      
  return color_img, gray_img


# function ในการ detection ใบหน้า
def face_detection_haarcascade(gray):
  
  # ค้นหาใบหน้า ต้องใช้รูปภาพสีเทาในการหา
  gray = np.array(gray, dtype='uint8')
  faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0)
  print('Detected Face:', len(faces))
  
  if(len(faces) == 0):
    x,y,w,h = [0,0,0,0]

  # วาดกรอบสี่เหลี่ยมรอบใบหน้า
  for (x,y,w,h) in faces:
    print("face location and size", x,y,w,h)
    
  return x,y,w,h

# ใช้ HoG ในการ detect ใบหน้า
def hog_feature(img):
  detected_face = cv2.resize(img, (224, 224))
  x = image.img_to_array(detected_face)
  x = np.expand_dims(x, axis=0)
  x = utils.preprocess_input(x, version=1)
  features = resnet50_features.predict(x)
  fd = np.array(features)
  #print(fd)
  return fd

# นำรูปภาพที่ดึงมาจาก loop อ่านรูปภาพจาก dir มาส่งค่าเข้าไปอ่านรูปและ detect ใบหน้า
for f in matches:
  label = f.split('/')[-1]
  labels = label.split('\\')[0]
  print('#', cnt, ':', labels)
  color_img, gray_img = return_image(f)
  x,y,w,h = face_detection_haarcascade(gray_img)
  
  if((x != 0) and (y != 0)):
    face_img = color_img[y:y+h, x:x+w]
    re_img = transform.resize(face_img, (n_rows, n_cols), anti_aliasing=False)
    fd = hog_feature(re_img)
    y_full_name.append(f)
    y_data.append(labels)
    x_data.append(fd)
  cnt+=1
  
x_data = np.asarray(x_data)

# รับรูปจากกล้อง webcam ด้วยการใส่ 0 เข้าไปในพารามิเตอร์
live = cv2.VideoCapture(0)

# เปิด while loop เพื่อใช้ในการอ่านรูปภาพจากตัวแปร live
while(True):
    r, f = live.read()
    loc = f
    color_live = np.array(loc, dtype='uint8')
    gray = cv2.cvtColor(loc, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0)

    for (x,y,w,h) in faces:
        color_face = cv2.rectangle(color_live,(x,y),(x+w,y+h),(0,255,0),4)
        crop_face = color_live[y:y+h, x:x+w]
        re_img = transform.resize(crop_face, (n_rows, n_cols), anti_aliasing=False)
        fd = hog_feature(re_img)

        #print("face location and size", x,y,w,h)
        #print('Similarity:', cosine_similarity(unknown_fv, known_fv)[0][0])
        for i in range(0, len(x_data)):
            unknown_fv = fd
            known_fv = x_data[i]
            simm_value = cosine_similarity(unknown_fv, known_fv)[0][0]
            if(simm_value >= thresh):
                simm.append(simm_value)
                simm_label.append(y_data[i])
                simm_label_fullpath.append(y_full_name[i])

                max_idx = simm.index(max(simm))
                lb = simm_label[max_idx].upper()
                print('Similar value:', simm[max_idx])

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_face, lb, (x+5,y-5), font, 1, (0,255,255), 2)
                cv2.putText(color_face, str(simm[max_idx]), (x+10,y+h-10), font, 1, (0,0,255), 2)
                cv2.imshow('My IP', color_face)
                  
    cv2.imshow('My IP', color_live)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
live.release()
cv2.destroyAllWindows()
