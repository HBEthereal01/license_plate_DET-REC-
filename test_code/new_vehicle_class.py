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

print(multiprocessing.cpu_count())
# img_out_name = f"./vehicle_result/crop_"
frame_out_name = f"F_"
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
    crop_count=1
    myframe = []
    config = { "apiKey": "AIzaSyDTqeM5PZd2mzBsYxzwQArkcFgRfUt4RlI","authDomain": "upload-images-aae3e.firebaseapp.com","projectId": "upload-images-aae3e","storageBucket": "upload-images-aae3e.appspot.com","messagingSenderId": "189694713606","appId": "1:189694713606:web:b993e8b9eb36f99b8e0887","measurementId": "G-GF5B84G1PN","serviceAccount": "serviceAccount.json","databaseURL": "https://console.firebase.google.com/project/upload-images-aae3e/database/upload-images-aae3e-default-rtdb/data/~2F"}

    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    print("-------------------Connected to Firebase-------------------- ")

    storage = firebase.storage()
    
    print("----- Searching for element-----") 
    while (True):
        
        pwd = os.getcwd()
        elements = os.listdir(pwd+"/vehicle_result")

        print("In loop ")
        # print(elements)
        if elements:
            for element in elements:
                # print(f"{frame_out_name}{crop_count}.jpg")
                print(f"{frame_out_name}{crop_count}.png")
                # if f"{crop_F}{crop_count}.png" == element:
                #     # storage.child(f"License_plate_{crop_count}.png").put(element)
                #     # print("saved")648.3896095752716
                #     # crop_count:  12


                #     myframe.append(element)
                #     print(element)
                #     lengthofstack = len(myframe)
                #     print("length of stack: ",lengthofstack)
                #     if lengthofstack == 5:
                #         timestamp1 = time.time()
                #         print("Stack is full")  
                        
                #         i=4
                #         pwd1 = os.getcwd()

                #         for idx in myframe:
                #             print("frame: ",idx)
                #             F_img_path = pwd1+"/result/"+idx
                #             crop_img_path = pwd1+"/result/"
                        
                #             print("License_plate_",crop_count-i)
                #             storage.child(f"{folder_name1}License_plate_frame{crop_count-i}.png").put(F_img_path)
                #             storage.child(f"{folder_name2}crop_image{crop_count-i}.png").put(f"{crop_img_path}{text_F}{crop_count-i}.png")
                #             storage.child(f"{folder_name3}text_image{crop_count-i}.txt").put(f"{crop_img_path}{text_F}{crop_count-i}.txt")   
                #             print("uploaded")

                #             print("Files deleted from folder")
                #             os.remove(F_img_path)
                #             os.remove(f"{crop_img_path}{text_F}{crop_count-i}.png")
                #             os.remove(f"{crop_img_path}{text_F}{crop_count-i}.txt")
                #             i=i-1



                #         print("Files deleted from folder")
                #         myframe=[]
                #         print("Image uploading successfully")
                #         timestamp2 = time.time()
                #         dt1 = timestamp2-timestamp1
                #         print("total time takes : "+str(dt1))
                    
                #     crop_count=crop_count+1 
                #     print("crop_count: ",crop_count)
                #     break
                        # if crop_count == 6:
                            # loop = False
                        # print("timestamp2 : ",timestamp2)           
                    # else:
                    #     crop_count=crop_count+1 
                    #     print("element is not found") 

        if not elements:
            print(" folder is empty")    
            print(elements)

def vehicle_classification():
    camera_url = "rtsp://admin:Dd22864549*@192.168.1.61:554/cam/realmonitor?channel=1&subtype=0"


    vid = cv2.VideoCapture(camera_url)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 10)

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

            frame,fps = put_FPS(frame)
            cv2.imshow("frame",frame)
            cv2.imwrite(f"{frame_out_name}{count}.png",frame)
            # storage.child(f"frame_{count}.jpg").put(f"{frame_out_name}{count}.jpg")
            print("Vehicle Count ="+str(vehicleCount)+" ,Person Count = "+str(personcount)+" ,motorcycle Count = "+str(motorcyclecount)+" ,Bicycle Count = "+str(bicyclecount))

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

if __name__ == "__main__":
    # firebase_upload()
    
    folder_name1 = "frame_result/"
    crop_F = "crop_F_"
    folder_name2 = "crop_result/"
    text_F = "crop_"
    folder_name3 = "text_result/"

   
    # p3 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name1,folder_name2,folder_name3,crop_F,text_F])
    p1 = multiprocessing.Process(target = ALPR_detection, args =[frame_out_name])
    p2 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name1,folder_name2,folder_name3,crop_F,text_F])
    # p2 = multiprocessing.Process(target= firebase_upload, args = [frame_out_name])

    print("program start !!!")
    p1.start()
    # print("firebase upload")
    p2.start()
    # p3.start()
    print("join")
    p1.join()
    p2.join()
    # p3.join()
    print("Program of ALPR detection and firebase uploading images are closed !!!!!!!!!!")
