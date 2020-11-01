from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import time

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# TensorFlow
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import shutil
import os, glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

framework="tf"  #tf, tflite, trt
model="yolov4"  #yolov3 or yolov4
tiny=False      #yolo or yolo-tiny
iou=0.45        #iou threshold
score=0.3      #score threshold
output='./detections/'  #path to output folder
weights_loaded="./checkpoints/custom-416/" #replace with your checkpoint

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

# hardcoded diction of classes as we have custome predefined classes
# class_dict = dict({0:"You are safe.",1:"You are not wearing a mask.",2:"Please wear your mask properly."})
lis=['Apple', 'Asparagus', 'Bagel', 'Banana', 'Bell pepper', 'Bread', 'Broccoli', 'Burrito', 'Cabbage', 'Cake', 'Candy', 'Cantaloupe', 'Carrot', 'Cheese', 'Common fig', 'Cookie', 'Crab', 'Croissant', 'Cucumber', 'Doughnut', 'Egg', 'French fries', 'Grape', 'Grapefruit', 'Guacamole', 'Honeycomb', 'Hot dog', 'Ice cream', 'Lemon', 'Lobster', 'Mango', 'Milk', 'Muffin', 'Mushroom', 'Orange', 'Oyster', 'Pancake', 'Pasta', 'Peach', 'Pear', 'Pineapple', 'Pizza', 'Pomegranate', 'Popcorn', 'Potato', 'Pretzel', 'Pumpkin', 'Radish', 'Salad', 'Shrimp', 'Squid', 'Strawberry', 'Sushi', 'Taco', 'Tart', 'Tomato', 'Waffle', 'Watermelon', 'Zucchini']

class_dict = dict()
for i in range(0,len(lis)):
    class_dict[i]=lis[i]


#def main():
def mask_detector(image_name):
    image_size=416
    imput_image=image_name
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = image_size
    images = [imput_image]

    # load model
    if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights_loaded)
    else:
            saved_model_loaded = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #image = utils.draw_bbox(original_image, pred_bbox)
        cropped_image = utils.draw_bbox(original_image, pred_bbox)
        # image = utils.draw_bbox(image_data*255, pred_bbox)
        image = Image.fromarray(cropped_image.astype(np.uint8))
        #if not FLAGS.dont_show:
            #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        cv2.imwrite(output + 'detection1.jpg', image)
        return image,classes.numpy()[0],scores.numpy()[0]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("********************************************************************************************************************")
        print(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
        image,classes,scores=mask_detector(file_path)
        cv2.imwrite('./static/detection.png', image)
        for i in range(0,len(scores)):
            if scores[i]==0:
                break
        classes = classes[:i]
        print(classes[:i])
        print("***************************************************")
        print(scores[:i])
        classes = np.unique(classes)
        # preds = model_predict(file_path, model)
        if classes.size == 0:
            result = "<br>Detected Nothing. Please enter a good quality image."
        else:
            result = "<br>"
        for c in range(0,len(classes)):
            if scores[c] >= 0.5:
                result = result + str(c+1) + '. ' + class_dict.get(classes[c]) + '&nbsp;&nbsp;<a href="#third_row" id="'+class_dict.get(classes[c])+'" class="custom-image stretched-link lead" style="color:#434445;font-size: 1.5rem;">(Get ' + class_dict.get(classes[c]) + ' Recipies)</a><br>'
        print(result)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result+"<br>"
        # return redirect(url_for('static', filename='detections/detection.png'), code=301)
        # return render_template('results.html')
    return None

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    # flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
    # flags.DEFINE_float('iou', 0.45, 'iou threshold')
    # flags.DEFINE_float('score', 0.25, 'score threshold')
    # flags.DEFINE_boolean('dont_show', False, 'dont show video output')
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    # video_path = 0

    # if FLAGS.framework == 'tflite':
    #     interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    #     interpreter.allocate_tensors()
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     print(input_details)
    #     print(output_details)
    # else:
    saved_model_loaded = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    
    vid = cv2.VideoCapture(0)

    out = None
    output = "D:/mask_web_app/static/detections"
    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # if FLAGS.framework == 'tflite':
        #     interpreter.set_tensor(input_details[0]['index'], image_data)
        #     interpreter.invoke()
        #     pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        #     if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
        #         boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
        #                                         input_shape=tf.constant([input_size, input_size]))
        #     else:
        #         boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
        #                                         input_shape=tf.constant([input_size, input_size]))
        # else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("result", result)
        
        if output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)

