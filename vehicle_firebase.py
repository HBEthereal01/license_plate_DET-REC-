import pyrebase
import os
import time
import multiprocessing


def firebase_upload_image (folder_name2,crop_F):
    global crop_count
    global timestamp1
    global timestamp2
    
    crop_count=1
    myframe = []
   
    config = {"apiKey": "AIzaSyBUBAHwXoKBX8cUsUBpxu2ViBjGCCkpnUk","authDomain": "vehicle-classification-d5904.firebaseapp.com","projectId": "vehicle-classification-d5904","storageBucket": "vehicle-classification-d5904.appspot.com","messagingSenderId": "431552048645","appId": "1:431552048645:web:1e5c97c3a540b44d21f66c","measurementId": "G-DY3BM732ZS","serviceAccount": "vehicleserviceAccount.json","databaseURL": "https://console.firebase.google.com/u/1/project/vehicle-classification-d5904/database/vehicle-classification-d5904-default-rtdb/data/~2F"}

    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    storage = firebase.storage()
    print("-------------------Connected to Firebase-------------------- ")
    print("----- Searching for element-----") 
    while (True):
        
        pwd = os.getcwd()
        elements = os.listdir(pwd+"/vehicle_result")

        print("In loop ")
        # print(elements)
        if elements:
            
            for element in elements:
                print(f"{crop_F}{crop_count}.png")
                # print(f"{crop_F}{crop_count}")
                if (f"{crop_F}{crop_count}.png") == element:
                    myframe.append(element)
                    lengthofstack = len(myframe)
                    print("length of stack: ",lengthofstack)
                    if lengthofstack == 5:
                        timestamp1 = time.time()
                        print("Stack is full")  
                        
                        i=4
                        pwd1 = os.getcwd()

                        for idx in myframe:
                            print("frame: ",idx)
                            F_img_path = pwd1+"/vehicle_result/"+idx
                            print("Vehicle_",crop_count-i)
                            storage.child(f"{folder_name2}vehicle_frame{crop_count-i}.png").put(F_img_path)
                            print("uploaded")
                            os.remove(F_img_path)
                            print("Files deleted from folder")
                            i=i-1
                        myframe=[]
                        print("Image uploading successfully")
                        timestamp2 = time.time()
                        dt1 = timestamp2-timestamp1
                        print("total time takes : "+str(dt1))
                    
                    crop_count=crop_count+1 
                    print("crop_count: ",crop_count)
                    break
                   
        else:
            print(" folder is empty")    
            print(elements)


if __name__ == "__main__":
    folder_name2 = "crop_result/"
    text_F = "crop_"
    p4 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name2,text_F])
    print("------------define--------------")
    p4.start()

    print("------------start------------")
    p4.join()
    print("-------------end--------------")