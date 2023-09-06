import cv2
import jetson_inference
from track_functions import  detectVehicles,opencvToCuda, cudaToOpencv, put_Rect, put_Text, put_FPS

camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.1.62:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink drop=1"
# camera_url = "rtsp://admin:Dd22864549*@192.168.1.62:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink drop=1"

vid = cv2.VideoCapture(camera_url)
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 10)

if not vid.isOpened():
    print("Error opening RTSP stream.")

img_out_name = f"./vehicle_result/crop_"

net = jetson_inference.detectNet(network="ssd-mobilenet-v2",threshold=0.5)    
count=0
while True:
    ret,frame = vid.read()

    if ret:
        frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height,width = frame.shape[0],frame.shape[1]
        cudaimg = opencvToCuda(frame)
        # detections= net.Detect(cudaimg,width,height)
        # print(detections)

        detections = detectVehicles(cudaimg,width,height,net)
        
        # vehicleCount=0
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
        cv2.imwrite(f"{img_out_name}{count}.png",frame)
        # storage.child(f"frame_{count}.jpg").put(f"{frame_out_name}{count}.jpg")

    if not ret:
        print("Error retrieving Frame")      
        break

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()     


