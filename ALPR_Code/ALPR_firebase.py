from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import datetime
import argparse
import function.helper as helper
import pyrebase
import multiprocessing

print(multiprocessing.cpu_count())
# img_out_name = f"./result/crop_"
frame_out_name = f"crop_F_"
crop_count=0

pwd = os.getcwd()
elements = os.listdir(pwd+"/result")

def firebase_upload_image (folder_name1,folder_name2,folder_name3,crop_F,text_F):
    global crop_count
    global timestamp1
    global timestamp2
    
    
    # print("folder name: ",folder_name)
    # print("frame name: ",frame_out_name)


    # loop = True
    crop_count=30
    myframe = []
    config = { "apiKey": "AIzaSyDTqeM5PZd2mzBsYxzwQArkcFgRfUt4RlI","authDomain": "upload-images-aae3e.firebaseapp.com","projectId": "upload-images-aae3e","storageBucket": "upload-images-aae3e.appspot.com","messagingSenderId": "189694713606","appId": "1:189694713606:web:b993e8b9eb36f99b8e0887","measurementId": "G-GF5B84G1PN","serviceAccount": "serviceAccount.json","databaseURL": "https://console.firebase.google.com/project/upload-images-aae3e/database/upload-images-aae3e-default-rtdb/data/~2F"}

    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    print("-------------------Connected to Firebase-------------------- ")

    storage = firebase.storage()
    
    print("----- Searching for element-----") 
    while (True):
        
        pwd = os.getcwd()
        elements = os.listdir(pwd+"/result")

        print("In loop ")
        # print(elements)
        if elements:
            print("searching.....")
            print(f"{crop_F}{crop_count}.png")
            lengthofelements = len(elements)
            
            count=1
            for element in elements:
                
                if f"{crop_F}{crop_count}.png" == element:
                    myframe.append(element)
                    print(element)
                    lengthofstack = len(myframe)
                    print("length of stack: ",lengthofstack)
                    if lengthofstack == 5:
                        timestamp1 = time.time()
                        print("Stack is full")  
                        
                        i=4
                        pwd1 = os.getcwd()

                        for idx in myframe:
                            print("frame: ",idx)
                            F_img_path = pwd1+"/result/"+idx
                            crop_img_path = pwd1+"/result/"
                        
                            print("License_plate_",crop_count-i)
                            storage.child(f"{folder_name1}License_plate_frame{crop_count-i}.png").put(F_img_path)
                            storage.child(f"{folder_name2}crop_image{crop_count-i}.png").put(f"{crop_img_path}{text_F}{crop_count-i}.png")
                            storage.child(f"{folder_name3}text_image{crop_count-i}.txt").put(f"{crop_img_path}{text_F}{crop_count-i}.txt")   
                            print("uploaded")

                            print("Files deleted from folder")
                            os.remove(F_img_path)
                            os.remove(f"{crop_img_path}{text_F}{crop_count-i}.png")
                            os.remove(f"{crop_img_path}{text_F}{crop_count-i}.txt")
                            i=i-1
                        print("Files deleted from folder")
                        myframe=[]
                        print("Image uploading successfully")
                        timestamp2 = time.time()
                        dt1 = timestamp2-timestamp1
                        print("total time takes : "+str(dt1))
                    
                    crop_count=crop_count+1 
                    print("crop_count: ",crop_count)
                    break
                else: 
                    count=count+1
                    # print("no. of elements presentin folder: ",lengthofelements)
                    # print("no. of count: ",count)
                    if count==lengthofelements:
                        break  


                  

        if not elements:
            print(" folder is empty")    
            print(elements)

    print("Error not uploading")  
    current_timestamp = datetime.datetime.now()
    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    with open("Uplaodinfailfile.txt", "w") as file:
        file.write(timestamp_str)
    print(timestamp_str)
    

def ALPR_detection(frame_out_name):
    # load model
    yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
    yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
    yolo_license_plate.conf = 0.60

    prev_frame_time = 0
    new_frame_time = 0

    #vid = cv2.VideoCapture("/dev/video0")
    #camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=0 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
    # camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0"
    #camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! appsink drop=1"
    camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.1.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.1.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx ! appsink drop=1"
    vid = cv2.VideoCapture(camera_url)


    # if not cam.isOpened():
    #         print("Error opening RTSP stream.")


    # vid = cv2.VideoCapture("1.mp4")
    img_out_name = f"./result/crop_"
    frame_out_name = f"./result/crop_F_"
    crop_count=0

    while(True):
        timestamp1 = time.time()
        ret,frame = vid.read()
        # frame = cv2.imread("./1.png")
        # cv2.imshow("Image",frame)
        
        if ret:
            frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            plates = yolo_LP_detect(frame)
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
                crop_img = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                #crop_count=crop_count+1
                cv2.imwrite(f"{img_out_name}img.png", crop_img)
                rc_image = cv2.imread(f"{img_out_name}img.png")
                
                # cv2.imwrite("crop.jpg", crop_img)
                # rc_image = cv2.imread("crop.jpg")
                lp = ""
                for cc in range(0,2):
                    for ct in range(0,2):
                        lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(rc_image, cc, ct))
                        #print("number_plate: ",lp)
                        # if lp == "unknown":
                        #     filename=f"{img_out_name}{crop_count}.txt"
                        #     outfile = open(filename,'w')
                        #     outfile.write(lp)

                        if lp != "unknown":
                            
                            crop_count=crop_count+1
                            list_read_plates.add(lp)
                            cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            print("number_plate: ",lp)
                            
                            cv2.imwrite(f"{img_out_name}{crop_count}.png", crop_img)
                            # storage.child(f"License_{count}.jpg").put(f"{img_out_name}{crop_count}.jpg")

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
            cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if lpp != "unknown":
                cv2.imwrite(f"{frame_out_name}{crop_count}.png",frame)
                # storage.child(f"License_{crop_count}.jpg").put(f"{frame_out_name}{crop_count}.jpg")
            timestamp2 = time.time()
            dt1 = timestamp2-timestamp1
            print("Total time takes to execute one frame : "+str(dt1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
            
        if not ret:
            print("Error retrieving frame.")  
            current_timestamp = datetime.datetime.now()
            timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            with open("timestamp.txt", "w") as file:
                file.write(timestamp_str)
            print(timestamp_str)
            break

        if not vid.isOpened():
            print("Error opening RTSP stream.")  
            break  

    vid.release()
    cv2.destroyAllWindows()     

# firebase_upload()


if __name__ == "__main__":
    # firebase_upload()
    
    folder_name1 = "frame_result/"
    crop_F = "crop_F_"
    folder_name2 = "crop_result/"
    text_F = "crop_"
    folder_name3 = "text_result/"

   
    # p3 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name1,folder_name2,folder_name3,crop_F,text_F])
    # p1 = multiprocessing.Process(target = ALPR_detection, args =[frame_out_name])
    p2 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name1,folder_name2,folder_name3,crop_F,text_F])
    # p2 = multiprocessing.Process(target= firebase_upload, args = [frame_out_name])

    print("program start !!!")
    # p1.start()
    # print("firebase upload")
    p2.start()
    # p3.start()
    print("join")
    # p1.join()
    p2.join()
    # p3.join()
    print("Program of ALPR detection and firebase uploading images are closed !!!!!!!!!!")
        
        