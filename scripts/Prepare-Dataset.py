import os
import cv2
import matplotlib.pyplot as plt

def preprocess_train_images(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img=cv2.resize(gray,(256,256))
    return resized_img


for item in os.listdir('../data/raw'):

    for c, img in enumerate(os.listdir("../data/raw/" + item)):
        
        img_path="../data/raw/"+item+'/'+img
        loaded_img=cv2.imread(img_path)
        processed_img=preprocess_train_images(loaded_img)
        saving_filename="item"+str(c)+".jpg"
        cv2.imwrite('../data/processed/'+item+"/"+saving_filename,processed_img)
