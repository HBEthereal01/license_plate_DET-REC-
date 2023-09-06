import pyrebase
import os
import time
import multiprocessing
import cv2

# print(multiprocessing.cpu_count())

# # frame_out_name = f"crop_F_"
# crop_count=0

# pwd = os.getcwd()
# elements = os.listdir(pwd+"/result")
# # if not elements:
#     print("element not found")
frame_out_name = f"./vehicle_result/F_"
def firebase_upload_image (folder_name1,frame_name):
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
        print(elements)
        if elements:
            for element in elements:
                # print(f"{frame_out_name}{crop_count}.jpg")
                print(f"{frame_name}{crop_count}.png")
                if f"{frame_name}{crop_count}.png" == element:
                    # storage.child(f"License_plate_{crop_count}.png").put(element)
                    # print("saved")

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
                            F_img_path = pwd1+"/vehicle_result/"+idx
                            # crop_img_path = pwd1+"/result/"
                        
                            print("License_plate_",crop_count-i)
                            storage.child(f"{folder_name1}frame{crop_count-i}.png").put(F_img_path)
                            # storage.child(f"{folder_name2}crop_image{crop_count-i}.png").put(f"{crop_img_path}{text_F}{crop_count-i}.png")
                            # storage.child(f"{folder_name3}text_image{crop_count-i}.txt").put(f"{crop_img_path}{text_F}{crop_count-i}.txt")   
                            print("uploaded")

                            print("Files deleted from folder")
                            os.remove(F_img_path)
                            # os.remove(f"{crop_img_path}{text_F}{crop_count-i}.png")
                            # os.remove(f"{crop_img_path}{text_F}{crop_count-i}.txt")
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
                    # if crop_count == 6:
                        # loop = False
                    # print("timestamp2 : ",timestamp2)           
                # else:
                #     print("element is not found") 
        else:
            print(" folder is empty")    
            print(elements)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

        # if element == None:
        #         print("folder is empty !!!")   
        # print("element in stack: ",myframe)

def firebase_upload_text (folder_name,frame_out_name,a):
    global crop_count
    global timestamp1
    global timestamp2
    
    
    print("folder name: ",folder_name)
    print("text name: ",frame_out_name)


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
        elements = os.listdir(pwd+"/result")

        print("In loop ",a)
        # print(elements)
        if elements:
            for element in elements:
                # print(f"{frame_out_name}{crop_count}.jpg")
                # print(element)
                if f"{frame_out_name}{crop_count}.txt" == element:
                    # storage.child(f"License_plate_{crop_count}.png").put(element)
                    # print("saved")

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
                            img_path = pwd1+"/result/"+idx
                            print("License_plate_",crop_count-i)
                            storage.child(f"{folder_name}{frame_out_name}{crop_count-i}.txt").put(img_path)   
                            i=i-1

                        for idx in myframe:
                            os.remove(pwd1+"/result/"+idx)

                        print("Files deleted from folder")
                        myframe=[]
                        print("Image uploading successfully")
                        timestamp2 = time.time()
                        dt1 = timestamp2-timestamp1
                        print("total time takes : "+str(dt1))
                    
                    crop_count=crop_count+1 
                    print("crop_count: ",crop_count)
                    break
                    # if crop_count == 6:
                        # loop = False
                    # print("timestamp2 : ",timestamp2)           
                # else:
                #     print("element is not found") 
        else:
            print(" folder is empty")    
            print(elements)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break     
            # if element == None:
            #     print("folder is empty !!!")   
        # print("element in stack: ",myframe)




if __name__ == "__main__":
    # firebase_upload()
    
    folder_name1 = "vehicle_result/"
    frame_name= "F_"
    # folder_name2 = "crop_result/"
    # text_F = "crop_"
    # folder_name3 = "text_result/"

    p1 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name1, frame_name])
    # p2 = multiprocessing.Process(target= firebase_upload_image, args = [folder_name2,frame_out_name2,2])
    # p3 = multiprocessing.Process(target= firebase_upload_text, args = [folder_name3,frame_out_name2,3])
    print("------------define--------------")

    p1.start()
    # p2.start()
    # p3.start()

    print("------------start------------")
    p1.join()
    # p2.join()
    # p3.join()
    print("-------------end--------------")














































# mylist= [1,2,3,4]

# square_result = multiprocessing.Array('i',4)
# cube_result = multiprocessing.Array('i',4)
# # square_sum= multiprocessing.Value('i')
# # cube_sum= multiprocessing.Value('j')

# def square_list (mylist, square_result):
    
#     for idx, num in enumerate(mylist):
#         square_result[idx]=num*num
#         print("result: ",square_result[idx])
#     square_sum=sum(square_result)
#     print("sqaure_sum: ",square_sum)

# def cube_list (mylist, cube_result):
#     for idx, num in enumerate(mylist):
#         cube_result[idx] = num*num*num
#         print("cude_result: ",cube_result[idx])
#     cube_sum=sum(cube_result)
#     print("cube sum: ",cube_sum)    

# if __name__ == '__main__':
#     p1 = multiprocessing.Process(target = square_list, args=[mylist, square_result])
#     p2 = multiprocessing.Process(target = cube_list, args=[mylist, cube_result])
    
#     start = time.time()
#     p1.start()
#     p2.start()

#     end = time.time()
#     dt = end-start
#     print("multiprocessor time: ",dt)
#     p1.join()
#     p2.join()


#     start = time.time()
#     square_list(mylist,square_result)
#     cube_list(mylist,cube_result)

#     end = time.time()
#     dt = end-start
#     print("simple time: ",dt)

#     # # print("sum of squares: ",square_sum)
#     # for i in range(4):
#     #     print(square_result[i])

#     # # print("sum of squares: ",cube_sum)
#     # for i in range(4):
#     #     print(cube_result[i])




