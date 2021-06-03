import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
import time

model = load_model('EmotionDetector.h5')

cap = cv2.VideoCapture(0)
while(True):
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier("C:/Users/aadis/OneDrive/Desktop/Explo/frontalFace.xml")
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40,40), flags=cv2.CASCADE_SCALE_IMAGE)
	x=10
	y=10
	w=0
	h=0
	org = (x,y+h+10)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.75
	color = (0,255,0)
	thickness = 2

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
		crop_img = frame[y:y+h, x:x+w]
        
		org = (x,y+h+10)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.75
		color = (0,255,0)
		thickness = 2
    
		dsize = (48, 48)

		crop_img = cv2.resize(crop_img, dsize)
		crop_img = crop_img.reshape(1, 48, 48, 3)
		prediction = model.predict(crop_img)
		res = np.argmax(prediction)
		num2em = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

		cv2.putText(frame, num2em[res], org, font, fontScale, color, thickness, cv2.LINE_AA)

	if(len(faces)==0):
		cv2.putText(frame, "No Face Detected", org, font, fontScale, color, thickness, cv2.LINE_AA)

	
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()