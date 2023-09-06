# import cv2
# import jetson_utils
# import numpy as np
# import time
import cv2
import jetson_utils
import numpy as np
from skimage.segmentation import clear_border
import pytesseract
import time
from PIL import Image
#import firebase_admin
#from firebase_admin import credentials
#from firebase_admin import db

#cred = credentials.Certificate('./wait-and-track-system-firebase-adminsdk-ggjw1-8c62434bca.json')
#firebase_admin.initialize_app(cred, {
#    'databaseURL': 'https://wait-and-track-system-default-rtdb.asia-southeast1.firebasedatabase.app/'
#})
#ref = db.reference('/Detected_plates')
##############################-----------------------------------------------Track Function


def sendToFirebase(number="-"):
    ref.push('License_Plate: '+number)

def camset():
	# gst-launch-1.0 rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! queue !  omxh264dec ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink

    # camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=480, height=360 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=640, height=480 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1280, height=720 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1920, height=1080 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink"
    # cam = cv2.VideoCapture(camera_url, cv2.CAP_GSTREAMER)
    
    cam = cv2.VideoCapture('/dev/video0')
    # cam.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    if not cam.isOpened():
        print("Error opening camera.")

    return cam


def camset2():
	# gst-launch-1.0 rtspsrc location=rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! queue !  omxh264dec ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink

    # camera_url = "rtsp://admin:Dd22864549*@10.13.1.60:554/cam/realmonitor?channel=1&subtype=0"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=480, height=360 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=640, height=480 ! appsink drop=1"
    camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! appsink drop=1"
    #camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@10.13.3.61:554 latency=0 ! application/x-rtp, media=video ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=1920, height=1080 ! appsink drop=1"
    # camera_url = "rtspsrc location=rtsp://admin:Dd22864549*@192.168.100.61:554 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! appsink"
    ##camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=1 ! rtph264depay ! h264parse ! nvv412decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480"
    # cam = cv2.VideoCapture(camera_url, cv2.CAP_GSTREAMER)
    #camera_url = "rtsp://admin:Dd22864549*@10.13.3.61:554/cam/realmonitor?channel=1&subtype=0 latency=0"

    cam = cv2.VideoCapture(camera_url)
    # cam.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    if not cam.isOpened():
        print("Error opening RTSP stream.")

    return cam

def put_Text(frame,text='NoText', x=10, y=10, font_scale=2, color=(0,0,255), text_thickness=1):

	if isinstance(text,float) or isinstance(text,int):
		text = str(round(text,2))
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_size = cv2.getTextSize(text,font, font_scale, text_thickness)[0]
	text_x = x + 10
	text_y = y + 15

	return cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, text_thickness)


timestamp = time.time()
fpsfilt=0
def put_FPS(frame):
	global timestamp, fpsfilt
	dt = time.time()-timestamp
	timestamp=time.time()
	fps=1/dt
	fpsfilt = 0.9*fpsfilt+0.1*fps
	
	text = 'FPS: '+str(round(fpsfilt,2))
	frame = put_Text(frame,text,x=5,y=10,font_scale=1,text_thickness=2)

	return frame,text



def put_Rect(img,top,left,bottom,right):
	green_color = (0,255,0)
	thickness = 1
	start_point = (left,top)
	end_point = (right,bottom)

	img = cv2.rectangle(img, start_point, end_point, green_color, thickness)

	return img

def opencvToCuda(frame):
    cudaimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    cudaimg = jetson_utils.cudaFromNumpy(cudaimg)
    return cudaimg

def cudaToOpencv(cudaimg,width,height):

    numpy_array = jetson_utils.cudaToNumpy(cudaimg,width,height,4)
    frame = cv2.cvtColor(numpy_array.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return frame

def reorientCudaimg(cudaimg,width,height,angle):

    frame = cudaToOpencv(cudaimg,width,height)

    h,w,c =  frame.shape
    center = (h/2,w/2)

    rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0)
    rotatedframe = cv2.warpAffine(frame,rotation_matrix,(w,h))

    return opencvToCuda(rotatedframe)


def detectVehicles(cudaimg,width,height,net):
    return net.Detect(cudaimg,width,height)


def detectPlate(cudaimg,width,height,net):

    opencvimg = cudaToOpencv(cudaimg,width,height)

    detections = net.Detect(cudaimg, width, height, overlay='lines,labels,conf')


    platelist=[]
    for detect in detections:
        left=detect.Left
        top=detect.Top
        bottom=detect.Bottom
        right=detect.Right
        confval = detect.Confidence
        area = detect.Area

        platelist.append([confval,area,top,bottom,left,right])
    
    sortedlist = sorted(platelist, key=lambda x:[x[1],x[0]], reverse=True)

    if len(sortedlist) == 0:
        return opencvimg,[0,0,0,0],0

    plate = sortedlist[0]

    top = plate[2]
    bottom = plate[3]
    left = plate[4]
    right = plate[5]

    plate_img = opencvimg[int(top):int(bottom),int(left):int(right)]

    return plate_img,[left,top,bottom,right],plate[0]


def recognizePlate(frame,net):
    height,width = frame.shape[0],frame.shape[1]

    cudaimg = opencvToCuda(frame)

    detections = net.Detect(cudaimg,width,height,overlay='lines,labels,conf')

    return cudaimg,width,height


def characterSegmentation(plate):
    grayplate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    
    # Sharpening
    gaussian_blur = cv2.GaussianBlur(grayplate,(7,7),10)
    sharpen = cv2.addWeighted(grayplate,3.5,gaussian_blur,-2.5,2)

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # White thickening
    thresh_dilate = cv2.dilate(thresh, rect_kern, iterations = 1)
    thresh_dilatecb = clear_border(thresh_dilate)

    # thresh_dilatecbinv = cv2.bitwise_not(thresh_dilate)
    thresh_dilatecbinv = cv2.bitwise_not(thresh_dilatecb)

    return thresh_dilatecbinv


def characterSegmentation2(plate):
    grayplate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    # grayplate = cv2.resize(grayplate, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    
    # Noise reduction
    # bilateral = cv2.bilateralFilter(grayplate, d=9, sigmaColor=75, sigmaSpace=75)
    

    # Sharpening
    gaussian_blur = cv2.GaussianBlur(grayplate,(7,7),10)
    sharpen = cv2.addWeighted(grayplate,3.5,gaussian_blur,-2.5,2)
    

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)



    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # White thickening
    thresh_dilate = cv2.dilate(thresh, rect_kern, iterations = 1)


    # Black thickening
    thresh_erode = cv2.erode(thresh,rect_kern, iterations = 1)
    
    
    # cv2.imshow('grayplate',grayplate)
    # cv2.imshow('sharpen',sharpen)
    # cv2.imshow('thresh',thresh)
    # cv2.imshow('thresh_dilate',thresh_dilate)
    # cv2.imshow('thresh_erode',thresh_erode)


    thresh_cb = clear_border(thresh)
    # cv2.imshow('thresh_cb',thresh_cb)

    thresh_dilatecb = clear_border(thresh_dilate)
    # cv2.imshow('thresh_dilatecb',thresh_dilatecb)

    # thresh_erodecb = clear_border(thresh_erode)
    # cv2.imshow('thresh_erodecb',thresh_erodecb)
    

    # thresh_cbinv = cv2.bitwise_not(thresh_cb)
    # cv2.imshow('thresh_cvinv',thresh_cbinv)

    thresh_dilatecbinv = cv2.bitwise_not(thresh_dilatecb)
    cv2.imshow('thresh_dilatecvinv',thresh_dilatecbinv)


    print(getNumberPlate(thresh_dilatecbinv))


  
def getNumberPlate(segmented_plate):
    #PSM(Page Segmentation Method) mode, Tesseract's setting has 14(0-13) modes of operation, 
    # psm 7 - treat the image as single text line
    # psm 8 - treat the image as a single word
    # psm 10 - treat the image as a single character
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(segmented_plate,config='-c tessedit_char_whitelist='+alphanumeric+' --psm 7 --oem 3')
    return text


def yolo_detector(model,img):
    
    frame = img
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
    results = model(image, size=640)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    classes = model.names

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    x_shape, y_shape = frame.shape[1], frame.shape[0]

    detections=[]
    for i in range(len(labels)):
        row = cordinates[i]
        left = int(row[0]*x_shape)
        top = int(row[1]*y_shape)
        right = int(row[2]*x_shape)
        bottom = int(row[3]*y_shape)
        conf = row[4]
        class_label = classes[int(labels[i])]
        
        detections.append([[top,left,bottom,right],conf,class_label])
            
    return detections



def haarcascade_detector(frame):

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    if plate_cascade.empty():
        print("Error: Cascade Classifier file not found or cannot be loaded.")
        return frame,[0,0,0,0]

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(plates) == 0:
        # print("Error: Number plate not detected.")
        return frame,[0,0,0,0]

    for (x,y,w,h) in plates:
        a,b = (int(0.02*frame.shape[0]), int(0.025*frame.shape[1]))

        plate = frame[y+a:y+h-a, x+b:x+w-b,:]

        # return plate,[(y+a), (x+b), (y+h-a), (x+w-b)]
        return plate,[(y), (x), (y+h), (x+w)]

def vehicles_det_count(frame):
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
            frame = put_Rect(frame,top,left,bottom,right)
            frame = put_Text(frame, item, x=left-5, y=top, font_scale=1.5, color=(0,255,0), text_thickness=2
            )
            text = 'Confidence: ' + str(round(conf,4)*100) + '%'
            frame = put_Text(frame, text, x=left-5, y=top+15, font_scale=0.5, color=(0,255,0), text_thickness=1)

    frame = put_Text(frame, 'Vehicle Count = '+str(vehicleCount), x=200, y=10, font_scale=1, text_thickness=2)
    frame,fps = put_FPS(frame)
    return frame