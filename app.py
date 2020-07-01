from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



map_category={0:"Chhoila",1:"Rice",2:"Kathi-Roll",3:"Laphing",4:"Mo:Mo",5:"Paani-Puri",
              6:"Yomari"}

with open("mobilenet02.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("trainStep2_mobilenet_weights.h5")


def model_predict(img_path):
    img=cv2.imread(img_path)
    processed_img=cv2.resize(img,(224,224))
    processed_img=processed_img.reshape(1,224,224,3)
    processed_img = processed_img / 255
    preds = loaded_model.predict(processed_img)
    indexPred = np.argmax(preds)
    return map_category[indexPred]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)