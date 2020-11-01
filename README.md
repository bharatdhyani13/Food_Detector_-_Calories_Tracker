# Covid_Safety_Detector_Yolov4
Face Detection using Yolov4 which tells weather a person is wearing a mask, not wearing a mask or wearing a mask but is not safe.

![](https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection4.png)

# Getting Started
Download all the prerequisits from requirement.txt or requirement-gpu.txt.

YOLOv4 comes pre-trained with weights of 80 different classes. Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste them into the 'data' folder of this repository. As the weight files are large, I haven't uploaded them here.

# Getting the data
Where to get the data in yolov4 format?

1 ) This has around 600 different classes of images available in yolov4 format.  https://storage.googleapis.com/openimages/web/index.html 

2 ) If you cannot find data in this format only other option is to make your own data like I did using labelimg : https://github.com/tzutalin/labelImg

I got the mask/no_mask data from https://www.kaggle.com/alexandralorenzo/maskdetection and improper_mask from [here](https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FIMFD&originalPath=aHR0cHM6Ly9lc2lnZWxlYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9jYWJhbmlfZXNpZ2VsZWNfZnIvRWlyalM4ZXc3LTVMbk84STU2VWs2M3dCS2Vid1NsdWtGQkZCYU84TjI1d24zZz9ydGltZT0tUF9ldm1CLTJFZw).

In order to train your own custom weights for a custom yolov4 object detector you need to follow this colab notebook : 

Get the custom weights after training and paste them into the 'data' folder of this repository.

## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/mask.jpg

# Run custom yolov4 model on video
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/masks.mp4 --output ./detections/results.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```

## Result on Images

<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection1.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection2.png" width="400">
<img src="https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/detection3.png" width="400">


## Result on Videos

See v.avi from here : (https://github.com/bharatdhyani13/Covid_Safety_Detector_Yolov4/blob/main/detections/)

# Sources
https://github.com/theAIGuysCode/tensorflow-yolov4-tflite

https://colab.research.google.com/drive/1-jJtE45jQfmA1a2Um4SD3HJAnB2N583R?usp=sharing
