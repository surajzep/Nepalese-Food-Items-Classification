import numpy as np
import os
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

def rotate(img):
    images=[img,img,img,img,img]
    rotate = iaa.Affine(rotate=(-60,60))         # rotate the images of input list with angle starting from -60 to +60
    images_rotated = rotate.augment_images(images)
    return images_rotated

def crop_flip_blur(img):
    images=[img,img,img,img,img]
    seq_1 = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))])
    modified_images=seq_1.augment_images(images)
    return modified_images


def augment_images(start_load_index,start_save_index):
    
    
    for item in (os.listdir("../data/processed/")):
        save_index=start_save_index
        
        for load_index in range(start_load_index,start_save_index):
            
            try:
            
                image=imageio.imread("../data/processed/"+item+"/item"+str(load_index)+".jpg")
                rotated_images=rotate(image)
                cropped_flipped_blurred=crop_flip_blur(image)
                
            except: 
                print("../data/processed/"+item+"/item"+str(load_index)+".jpg could not be read!")
            
            
            for rotated_image in rotated_images:
                
                
                imageio.imwrite(("../data/processed/"+item+"/item"+str(save_index)+".jpg"),rotated_image )
                save_index=save_index+1
                
                
            for cropped_flipped_blurred_image in cropped_flipped_blurred:
                
                saving_filename="item"+str(save_index)+".jpg"
                imageio.imwrite(("../data/processed/"+item+"/item"+str(save_index)+".jpg"),
                                cropped_flipped_blurred_image )
                save_index=save_index+1    
                   


augment_images(0, 51)
augment_images(51,561)                    