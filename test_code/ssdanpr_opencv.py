import cv2
import jetson_inference
import jetson_utils
import numpy as np
from wait_functions import camset, detectPlate, put_Rect, put_FPS, put_Text, characterSegmentation, getNumberPlate
#from wait_functions import sendToFirebase
from datetime import datetime

cam = camset()

#net = jetson_inference.detectNet(model='./weights_plate/plate_ssdmobilenetv1.onnx', labels='./weights_plate/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.5)
net = jetson_inference.detectNet(model='./weight_plate_new1/ssd-mobilenet.onnx', labels='./weight_plate_new1/labels.txt', input_blob='input_0', output_cvg='scores', output_bbox='boxes', threshold=0.5)
	
while True:
	ret, frame = cam.read()

	if ret:
		height,width = frame.shape[0],frame.shape[1]	
		cudaimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
		cudaimg = jetson_utils.cudaFromNumpy(cudaimg)

		plate,coords,conf = detectPlate(cudaimg,width,height,net)
		left = int(coords[0])
		top = int(coords[1])
		bottom = int(coords[2])
		right = int(coords[3])

		# plate = cv2.resize(plate,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)


		# preprocessed_plate = characterSegmentation(plate)

		# number = getNumberPlate(preprocessed_plate)

		
		frame = put_Rect(frame,top,left,bottom,right)
		frame = put_Text(frame,conf,left,bottom,1,(0,0,255),2)

		# preprocessed_plate,fps = put_FPS(preprocessed_plate)
		frame,fps = put_FPS(frame)

		cv2.imshow('SSD_Plate_Detector_Opencv',frame)

		# current_time = datetime.now().strftime("%H:%M:%S")
		# print('License plate:', number ,fps, 'Time:',current_time, 'Frame:',frame.shape)

	if not ret:
		print("Error retrieving frame.")
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
