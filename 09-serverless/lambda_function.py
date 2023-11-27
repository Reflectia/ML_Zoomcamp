import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Preparing the image

from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img):
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X /= 255  
    return X


#url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'


def predict(url):
    img = download_image(url)
    img = prepare_image(img, (150,150))
    X = preprocess_image(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0].tolist()


def lambda_handler(event, context):
    url = event['url']
    
    result = predict(url)
    return result