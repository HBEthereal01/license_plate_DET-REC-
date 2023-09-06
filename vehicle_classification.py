from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper
import jetson_inference
import jetson_utils
import numpy as np
from track_functions import  detectVehicles,opencvToCuda, cudaToOpencv, put_Rect, put_Text, put_FPS
import pyrebase
import datetime


# config = {
#     "apiKey": "AIzaSyDTqeM5PZd2mzBsYxzwQArkcFgRfUt4RlI",
#   "authDomain": "upload-images-aae3e.firebaseapp.com",
#   "projectId": "upload-images-aae3e",
#   "storageBucket": "upload-images-aae3e.appspot.com",
#   "messagingSenderId": "189694713606",
#   "appId": "1:189694713606:web:b993e8b9eb36f99b8e0887",
#   "measurementId": "G-GF5B84G1PN",
#   "serviceAccount": "serviceAccount.json",
#   "databaseURL": "https://console.firebase.google.com/project/upload-images-aae3e/database/upload-images-aae3e-default-rtdb/data/~2F"
# }

# # load model
# yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
# yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
# yolo_license_plate.conf = 0.60

# prev_frame_time = 0
# new_frame_time = 0

#vid = cv2.VideoCapture("/dev/video0")
#camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
#camera_url = "rtsp://admin:Dd22864549*@10.13.3.62:554/cam/realmonitor?channel=1&subtype=0"
#camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
# camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.62:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! appsink drop=1"
camera_url = "rtsp://admin:Dd22864549*@192.168.1.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink drop=1"
# camera_url = "rtsp://admin:Dd22864549*@192.168.1.61:554/cam/realmonitor?channel=1&subtype=0"
        

vid = cv2.VideoCapture(camera_url)
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 10)

# vid = cv2.VideoCapture("1.mp4")
# img_out_name = f"./result/crop_"
# frame_out_name = f"./result/crop_F_"
# crop_count=0

net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)

# firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
frame_out_name = f"./vehicle_result/F_"
count=0

while True:
    ret,frame = vid.read()
    # frame2 = frame
    if ret:
        height,width = frame.shape[0],frame.shape[1]
        cudaimg = opencvToCuda(frame)

        detections = detectVehicles(cudaimg,width,height,net)

        vehicleCount = 0
        personcount = 0
        motorcyclecount = 0
        bicyclecount = 0

        for detect in detections:
            ID = detect.ClassID
            item = net.GetClassDesc(ID)
            conf = detect.Confidence
            top, bottom, left, right = int(detect.Top), int(detect.Bottom), int(detect.Left), int(detect.Right)
            if ID>=1 and ID<8:
                if ID == 1:
                    personcount=personcount+1
                elif ID == 4:
                    motorcyclecount= motorcyclecount+1
                elif ID == 2:
                    bicyclecount= bicyclecount+1
                else:
                    vehicleCount= vehicleCount+1

                count+=1
                frame = put_Rect(frame,top,left,bottom,right)
                frame = put_Text(frame, item, x=left-5, y=top, font_scale=1.5, color=(0,255,0), text_thickness=2)
                text = 'Confidence: ' + str(round(conf,4)*100) + '%'
                frame = put_Text(frame, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

        frame = put_Text(frame, 'Vehicle Count = '+str(vehicleCount+motorcyclecount+bicyclecount), x=200, y=10, font_scale=1, text_thickness=2)
        #frame = put_Text(frame, 'Person Count = '+str(personcount), x=200, y=10, font_scale=1, text_thickness=2)
        # frame = put_Text(frame, 'Bicycle Count = '+str(bicyclecount), x=200, y=10, font_scale=1, text_thickness=2)
        # frame = put_Text(frame, 'Motorcycle Count = '+str(motorcyclecount), x=200, y=10, font_scale=1, text_thickness=2)

        frame,fps = put_FPS(frame)
        cv2.imshow("frame",frame)
        cv2.imwrite(f"{frame_out_name}{count}.png",frame)
        # storage.child(f"frame_{count}.jpg").put(f"{frame_out_name}{count}.jpg")
        # print("Vehicle Count ="+str(vehicleCount)+" ,Person Count = "+str(personcount)+" ,motorcycle Count = "+str(motorcyclecount)+" ,Bicycle Count = "+str(bicyclecount))

    if not ret:
        print("Error retrieving Frame")   
        current_timestamp = datetime.datetime.now()
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with open("timestamp.txt", "w") as file:
          file.write(timestamp_str)
        print(timestamp_str)
        break

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()