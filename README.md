# Food_Detector_and_Calories_Tracker
This is a food detection app made in python flask which can detect food by providing any image of it. It can detect upto 59 selective classes of food on which the algorithm is trained on. We use YoloV4 to detect images.

![](https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/detection3.png)

# Getting the data
Where to get the data in yolov4 format?

1 ) This has around 600 different classes of images available in yolov4 format (I used this link to get 59 classes of food products).  https://storage.googleapis.com/openimages/web/index.html 

2 ) If you cannot find data in this format only other option is to make your own data using labelimg : https://github.com/tzutalin/labelImg

In order to train your own custom weights for a custom yolov4 object detector you need to follow this colab notebook : 

Get the custom weights after training and paste them into the 'data' folder of this repository.

## YOLOv4 Using Tensorflow (tf, .pb model)



## Result on Images

<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection1.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection2.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection3.png" width="400">


## Result on Videos

See v.avi from here : (https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/)

# Sources
https://github.com/theAIGuysCode/tensorflow-yolov4-tflite

https://colab.research.google.com/drive/1-jJtE45jQfmA1a2Um4SD3HJAnB2N583R?usp=sharing
