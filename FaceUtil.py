import cv2
import numpy as np
import os
from cv2 import face
from PIL import Image
from ginilib.Search import ColorDescriptor
from ginilib import faceutil
import imutils
import dlib
import urllib2
import sqlite3

class faces:
	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('./models/haarcascades/haarcascade_frontalface_default.xml')
		self.recognizer=cv2.face.createLBPHFaceRecognizer() 
		self.recognizer.load("./models/gini/trainingData.yml")
		self.font= cv2.FONT_HERSHEY_SIMPLEX
	
	def getid(self):
		con= sqlite3.connect("users.db")
		cur=con.execute("select max(id) from users")
		for row in cur:
			x=row[0]
			print x+1
		
		con.close()
		return x+1

	def add_User(self):
		cap = cv2.VideoCapture(0)
		samplenum=0
		ids=None
		self.recognizer.setThreshold(100)
		while(True):
		# Capture frame-by-frame
			ret,img=cap.read()
			# Our operations on the frame come here
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# Show faces
			faces=self.face_cascade.detectMultiScale(img,1.3,5,minSize=(30,30))

			for(x,y,w,h) in faces:
				x = x+ (0.10*w)
                                w = 0.8* w
                                h = 0.95*h
				x= int(x)
				w=int(w)
				h=int(h)
				samplenum = samplenum +1
				image = gray[y:y+h,x:x+w]
				dst = np.zeros(shape=(5,2))
				norm_image = cv2.normalize(image,dst,0,255,cv2.NORM_MINMAX)
				id,conf=self.recognizer.predict(norm_image)
				#print id , conf	
				if id==-1 :
                     			if ids==None:
						ids=self.getid() #get new id
					cv2.imwrite("img/users/" + str(ids) + "." + str(samplenum) + ".jpg" , norm_image)
				else:
                			cv2.putText(gray,str(id),(x,y+h),self.font,2,255)


				cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
	        		cv2.waitKey(100)
				cv2.imshow("Face",gray)

			cv2.waitKey(1)
			if(samplenum >20):
				break
	 	self.train()
		cap.release()
		cv2.destroyAllWindows() 
		return

	def train(self):
		path="img/users/"
		imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
		faces=[]
		ids=[]
		for imagePath in imagePaths:
			faceImg=Image.open(imagePath).convert('L');
			faceNP=np.array(faceImg,"uint8")
                	id= int(os.path.split(imagePath)[-1].split('.')[0])
                	print id
			faces.append(faceNP)
			ids.append(id)
			cv2.imshow("Training", faceNP)
			cv2.waitKey(20)

		self.recognizer.train(faces,np.array(ids))
		self.recognizer.save("models/gini/trainingData.yml")
		self.recognizer.load("./models/gini/trainingData.yml")
		cv2.destroyAllWindows	
		return 



	def detect(self):
		cap = cv2.VideoCapture(0)
		cd=ColorDescriptor()
		self.recognizer.setThreshold(100)
		while(True):
		# Capture frame-by-frame
			ret,img=cap.read()
		# Our operations on the frame come here
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			
		# Show faces
			faces=self.face_cascade.detectMultiScale(gray,1.3,5,minSize=(30,30))
			for(x,y,w,h) in faces:
				x = x+ (0.1*w)
                                w = 0.8* w
                                h = 0.95*h
				x= int(x)
				w=int(w)
				h=int(h)

				cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
				dst = np.zeros(shape=(5,2))
				norm_image1 = cv2.normalize(gray[y:y+h,x:x+w],dst,0,255,cv2.NORM_MINMAX)
				id,conf=self.recognizer.predict(norm_image1)
				#print id , conf	
				if id==-1 :
					cv2.putText(gray,"unknown",(x,y+h),self.font,2,255)	
				else:
                			cv2.putText(gray,str(id),(x,y+h),self.font,2,255)
						
						
	        		cv2.waitKey(100)
				cv2.imshow("Face",gray)
				id=0
			key = cv2.waitKey(1) & 0xFF
 
			# if the `q` key is pressed, break from the lop
			if key == ord("q"):
				break
		cap.release()
		cv2.destroyAllWindows() 
		return
          
	
	
