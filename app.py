from flask import Flask, redirect,render_template,request,send_from_directory, url_for
import pickle
import os
import numpy as np


import os
import sys


from flask import Flask, request, render_template, Response, jsonify

from gevent.pywsgi import WSGIServer


import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

import keras.utils as image
from werkzeug.utils import secure_filename

import numpy as np

# Declare a flask app
app = Flask(__name__)
MODEL_PATH = 'models/Detect1_Pneumonia.h5'

model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

# Predicting and preprocessing the image
def model_predict(img, model):
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    
    preds = model.predict(x)
    print(preds)
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ans=""
    if request.method == 'POST':
        
        # Get the image from post request
        f = request.files['file']

        # Save the file to .static/uploads
        basepath = os.path.dirname(__file__)
        name = "img.jpg"
        file_path = os.path.join(basepath, 'static/uploads', name)
        f.save(file_path)

        
        img = image.load_img(file_path, target_size=(150,150))

        preds = model_predict(img, model)
        
        
        result = preds[0,0]
        
        print(result)
        if result>0.5:
      
           ans='PNEUMONIA' 
        else:
     
            ans='NORMAL'   
        
    return render_template("result.html",ans=ans) 

if __name__ == "__main__":
    app.run(debug=True)

