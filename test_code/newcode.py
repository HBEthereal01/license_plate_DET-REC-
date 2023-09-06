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

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0

#vid = cv2.VideoCapture("/dev/video0")
#camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
#camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0"
#camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! appsink drop=1"

vid = cv2.VideoCapture(camera_url)

# vid = cv2.VideoCapture("1.mp4")
img_out_name = f"./result/crop_"
frame_out_name = f"./result/crop_F_"
crop_count=0

net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)

while True:
    ret,frame = vid.read()
    frame2 = frame
    if ret:
        height,width = frame.shape[0],frame.shape[1]
        cudaimg = opencvToCuda(frame)

        detections = detectVehicles(cudaimg,width,height,net)

        vehicleCount = 0
        for detect in detections:
            ID = detect.ClassID
            item = net.GetClassDesc(ID)
            conf = detect.Confidence
            top, bottom, left, right = int(detect.Top), int(detect.Bottom), int(detect.Left), int(detect.Right)
            if ID>=1 and ID<8:
                vehicleCount+=1
                frame1 = put_Rect(frame,top,left,bottom,right)
                frame1 = put_Text(frame1, item, x=left-5, y=top, font_scale=1.5, color=(0,255,0), text_thickness=2)
                text = 'Confidence: ' + str(round(conf,4)*100) + '%'
                frame1 = put_Text(frame1, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

        frame1 = put_Text(frame1, 'Vehicle Count = '+str(vehicleCount), x=200, y=10, font_scale=1, text_thickness=2)
        frame1,fps = put_FPS(frame1)
        

        plates = yolo_LP_detect(frame2)
        #plates = yolo_LP_detect(frame)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        lpp = ""
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = frame2[y:y+h, x:x+w]
            frame2 = cv2.rectangle(frame2, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            #crop_count=crop_count+1
            cv2.imshow("croped_img",crop_img)
            cv2.imwrite(f"{img_out_name}img.jpg", crop_img)
            rc_image = cv2.imread(f"{img_out_name}img.jpg")
            
            # cv2.imwrite("crop.jpg", crop_img)
            # rc_image = cv2.imread("crop.jpg")
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    #print("number_plate: ",lp)
                    # if lp == "unknown":
                    #     filename=f"{img_out_name}{crop_count}.txt"
                    #     outfile = open(filename,'w')
                    #     outfile.write(lp)

                    if lp != "unknown":
                        lpp = lp
                        crop_count=crop_count+1
                        list_read_plates.add(lp)
                        cv2.putText(frame2, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        print("number_plate: ",lp)
                        cv2.imwrite(f"{img_out_name}{crop_count}.jpg", crop_img)
                        filename=f"{img_out_name}{crop_count}.txt"
                        outfile = open(filename,'w')
                        outfile.write(lp)
                        flag = 1
                        break
                if flag == 1:
                    break
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame2, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('display',frame2)
        #cv2.imshow("frame",frame1)
        
        # if lpp != "unknown":
        #     #return frame
        #     cv2.imwrite(f"{frame_out_name}{crop_count}.jpg",frame)


    if not ret:
        print("Error retrieving Frame")    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

# def vehicles_det_count(frame):
#       print("hello vehicles")


#     height,width = frame.shape[0],frame.shape[1]
#     cudaimg = opencvToCuda(frame)

#     detections = detectVehicles(cudaimg,width,height,net)

#     vehicleCount = 0
#     for detect in detections:
#         ID = detect.ClassID
#         item = net.GetClassDesc(ID)
#         conf = detect.Confidence
#         top, bottom, left, right = int(detect.Top), int(detect.Bottom), int(detect.Left), int(detect.Right)
#         if ID>=1 and ID<8:
#             vehicleCount+=1
#             frame = put_Rect(frame,top,left,bottom,right)
#             frame = put_Text(frame, item, x=left-5, y=top, font_scale=1.5, color=(0,255,0), text_thickness=2
#             )
#             text = 'Confidence: ' + str(round(conf,4)*100) + '%'
#             frame = put_Text(frame, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

#     frame = put_Text(frame, 'Vehicle Count = '+str(vehicleCount), x=200, y=10, font_scale=1, text_thickness=2)
#     frame,fps = put_FPS(frame)
#     return frame


# def Alpr_det(frame,yolo_LP_detect,yolo_license_plate):
#     print("hello .....")

#     plates = yolo_LP_detect(frame)
#     list_plates = plates.pandas().xyxy[0],values.tolist()
#     list_read_plates = set()
#     lpp=""

#     for plate in list_plates:
#         flag = 0
#         x = int(plate[0]) # xmin
#         y = int(plate[1]) # ymin
#         w = int(plate[2] - plate[0]) # xmax - xmin
#         h = int(plate[3] - plate[1]) # ymax - ymin  
#         crop_img = frame[y:y+h, x:x+w]
#         cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
#         #crop_count=crop_count+1
#         cv2.imwrite(f"{img_out_name}img.jpg", crop_img)
#         rc_image = cv2.imread(f"{img_out_name}img.jpg")
            
#             # cv2.imwrite("crop.jpg", crop_img)
#             # rc_image = cv2.imread("crop.jpg")
#         lp = ""
#         for cc in range(0,2):
#             for ct in range(0,2):
#                 lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
#                     #print("number_plate: ",lp)
#                     # if lp == "unknown":
#                     #     filename=f"{img_out_name}{crop_count}.txt"
#                     #     outfile = open(filename,'w')
#                     #     outfile.write(lp)

#                 if lp != "unknown":
#                     lpp = lp
#                     crop_count=crop_count+1
#                     list_read_plates.add(lp)
#                     cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                     print("number_plate: ",lp)
#                     cv2.imwrite(f"{img_out_name}{crop_count}.jpg", crop_img)
#                     filename=f"{img_out_name}{crop_count}.txt"
#                     outfile = open(filename,'w')
#                     outfile.write(lp)
#                     flag = 1
#                     break
#             if flag == 1:
#                 break
#     new_frame_time = time.time()
#     fps = 1/(new_frame_time-prev_frame_time)
#     prev_frame_time = new_frame_time
#     fps = int(fps)
#     cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
#     # cv2.imshow('frame', frame)
#     if lpp != "unknown":
#         return frame





