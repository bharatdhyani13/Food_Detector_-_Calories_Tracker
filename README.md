# Food_Detector_and_Calories_Tracker
This is a food detection app made in python flask which can detect food by providing any image of it. It can detect upto 59 selective classes of food on which the algorithm is trained on. We use YoloV4 to detect images.

![](https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/detection3.png)

# Getting the data
Where to get the data in yolov4 format?

1 ) This has around 600 different classes of images available in yolov4 format (I used this link to get 59 classes of food products).  https://storage.googleapis.com/openimages/web/index.html 

2 ) If you cannot find data in this format only other option is to make your own data using labelimg : https://github.com/tzutalin/labelImg

In order to train your own custom weights for a custom yolov4 object detector you need to follow this colab notebook : 

Get the custom weights after training and paste them into the 'data' folder of this repository.

## Workflow
1) Go to the webpage : 
<img src="https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/c.png" width="200">
2) Now select Choose Image option and browse a clear image of your food: 
<img src="https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/choose.png" width="200">
3) Click on the Predict Button and wait for 10-15 seconds for the algorithm to run its predictions: 
<img src="https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/predict.png" width="200">
4) A pop up will apear with the predicted image, you can now choose the option to get various recipies for the food that was predicted : 
<img src="https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/get_recipe.png" width="200">
5) The cards will give you the exact Kcal count of the recipe with their most essential nutrients content : 
<img src="https://github.com/bharatdhyani13/Food_Detector_and_Calories_Tracker/blob/main/detections/recipe.png" width="200">

## Result on Images

<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection1.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection2.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection3.png" width="400">


## Result on Videos

See v.avi from here : (https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/)

# Sources
https://github.com/theAIGuysCode/tensorflow-yolov4-tflite

https://colab.research.google.com/drive/1-jJtE45jQfmA1a2Um4SD3HJAnB2N583R?usp=sharing
