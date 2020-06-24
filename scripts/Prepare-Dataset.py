import os
import cv2
import matplotlib.pyplot as plt


def preprocess_train_images(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img=cv2.resize(gray,(256,256))
    return resized_img


parent_dir="../data"
processed_dir="processed"
processed_path=os.path.join(parent_dir,processed_dir)
try:
    os.mkdir(processed_path)
    
except OSError as error:
    print(error)

train_dir="train"
test_dir="test"

processed_train_path=os.path.join(processed_path,train_dir)
processed_test_path=os.path.join(processed_path,test_dir)

try:
    os.mkdir(processed_train_path)
    
except OSError as error:
    print(error)
    
try:
    os.mkdir(processed_test_path)
    
except OSError as error:
    print(error)    

    
for item in os.listdir('../data/raw/train'):
    parent_dir="../data/processed/train"
    path=os.path.join(parent_dir,item)
    try:
        os.mkdir(path)
        
    except OSError as error:
        print(error)
        
    for c,img in enumerate(os.listdir('../data/raw/train/'+item)):
        img_path="../data/raw/train/"+item+'/'+img
        loaded_img=cv2.imread(img_path)
        processed_img=preprocess_train_images(loaded_img)
        saving_filename="item"+str(c)+".jpg"
        cv2.imwrite(os.path.join(path,saving_filename),processed_img)
        
        
        
        
for item in os.listdir('../data/raw/test'):
    parent_dir="../data/processed/test"
    path=os.path.join(parent_dir,item)
    try:
        os.mkdir(path)
        
    except OSError as error:
        print(error)
        
    for c,img in enumerate(os.listdir('../data/raw/test/'+item)):
        img_path="../data/raw/test/"+item+'/'+img
        loaded_img=cv2.imread(img_path)
        processed_img=preprocess_train_images(loaded_img)
        saving_filename="item"+str(c)+".jpg"
        cv2.imwrite(os.path.join(path,saving_filename),processed_img)        