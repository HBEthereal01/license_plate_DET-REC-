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

# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.8

prev_frame_time = 0
new_frame_time = 0

#vid = cv2.VideoCapture("/dev/video0")
#camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"

# camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"

# vid = cv2.VideoCapture(camera_url)

# vid = cv2.VideoCapture("1.mp4")
#while(True):
#ret,frame = vid.read()

img_out_name = f"./result/crop_"

frame = cv2.imread("./8.png")
cv2.imshow("Image",frame)
#if ret:
plates = yolo_LP_detect(frame)
#print("Plates: ",plates)
#plates = yolo_LP_detect(frame)
list_plates = plates.pandas().xyxy[0].values.tolist()
#print("list_plates: ",list_plates)
list_read_plates = set()
#print("list_read_plates: ",list_read_plates)
for plate in list_plates:
    flag = 0
    x = int(plate[0]) # xmin
    #print("X: ",x)
    y = int(plate[1]) # ymin
    #print("y: ",y)
    w = int(plate[2] - plate[0]) # xmax - xmin
    #print("w: ",x)
    h = int(plate[3] - plate[1]) # ymax - ymin  
    #print("h: ",x)
    crop_img = frame[y:y+h, x:x+w]
   
    cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
    cv2.imwrite(f"{img_out_name}8.jpg", crop_img)
    cv2.imshow(f"{img_out_name}8.jpg",crop_img)

    rc_image = cv2.imread("crop.jpg")
    #print('rc_image: ',rc_image)
    lp = ""
    for cc in range(0,2):
        for ct in range(0,2):
            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
            print("number_plate: ",lp)
            filename=f"{img_out_name}8.txt"
            outfile = open(filename,'w')
            outfile.write(lp)
            #cv2.imwrite(f"{img_out_name}3.txt", str(lp))


            if lp != "unknown":
                list_read_plates.add(lp)
                cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                flag = 1
                break
        if flag == 1:
            break
new_frame_time = time.time()
fps = 1/(new_frame_time-prev_frame_time)
prev_frame_time = new_frame_time
fps = int(fps)
cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
cv2.imshow('frame', frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break    

cv2.waitKey(0)    
# if not ret:
    #     print("Error retrieving frame.") 

#vid.release()
cv2.destroyAllWindows()           