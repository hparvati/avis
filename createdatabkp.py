from ginilib.Search import ColorDescriptor
from ginilib import util
from similiratyMeasure import Similarity

import cv2
import numpy as np
import os
from cv2 import face
from PIL import Image
import imutils
import dlib
import urllib2
import csv
import math
import shutil
import sqlite3

class faces:
	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('./models/haarcascades/haarcascade_frontalface_default.xml')
		self.recognizer=cv2.face.createLBPHFaceRecognizer() 
		self.recognizer.load("./models/gini/trainingData.yml")
		self.indexPath = "models/gini/index.csv"
		self.font= cv2.FONT_HERSHEY_SIMPLEX
		self.datapath="models/gini/hogface/shape_predictor_68_face_landmarks.dat"
		self.sm= Similarity()
		
		
	
	def getid(self):
		con= sqlite3.connect("users.db")
		cur=con.execute("select max(id) from users")
		for row in cur:
			x=row[0]
			print x+1
		
		con.close()

		return x+1

	def InsertUSer(self,uid,name,age,address):
		con= sqlite3.connect("users.db")
		cur=con.cursor()
		cur.execute('''INSERT INTO users(ID,NAME,AGE,ADDRESS) VALUES(?,?,?,?)''',(uid,name,age,address))
		con.commit()		
		con.close()
		return 

	def detect(self):
		cap = cv2.VideoCapture(0)
		self.recognizer.setThreshold(100)
		cd=ColorDescriptor()
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
        

	def detect_dlib(self):
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(self.datapath)
		cap = cv2.VideoCapture(0)
		samplenum=0
		self.recognizer.load("./models/gini/trainingData.yml")	
		self.recognizer.setThreshold(100)

		while(True):
			# Capture frame-by-frame
			ret,img=cap.read()
	
			# Our operations on the frame come here
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			gray=np.array(gray,"uint8")
			# Show faces
			rects = detector(gray, 1)
                     
			for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
				shape = predictor(gray, rect)
				shape = faceutil.shape_to_np(shape)
 		
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = faceutil.rect_to_bb(rect)
				cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
				dst = np.zeros(shape=(5,2))
				norm_image1 = cv2.normalize(gray[y:y+h,x:x+w],dst,0,255,cv2.NORM_MINMAX)
				norm_image1=self.ImageProcess(norm_image1)
				id,conf=self.recognizer.predict(norm_image1)
				print id , conf	
				if id==-1 :
					cv2.putText(gray,"unknown",(x,y+h),self.font,2,255)	
					cv2.imwrite("img/users/" + str(samplenum) + ".jpg" , norm_image1)
					samplenum=samplenum+1
					res=self.describenew(shape)
					print res

				else:
                			cv2.putText(gray,str(id),(x,y+h),self.font,2,255)
					
			
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
				for (a, b) in shape:
					cv2.circle(gray, (a, b), 1, (0, 0, 255), -1)
					
 
			# show the output image with the face detections + facial landmarks
				dst = np.zeros(shape=(5,2))
				norm_image1 = cv2.normalize(gray[y:y+h,x:x+w],dst,0,255,cv2.NORM_MINMAX)
				cv2.imshow("Output", gray)
		
			key = cv2.waitKey(1) & 0xFF
 
			# if the `q` key is pressed, break from the lop
			if key == ord("q"):
				break
		cap.release()
		cv2.destroyAllWindows() 
		return


	def BulkImageLoad(self):
		path="img/users/"
		processed="img/processed/"
		unprocessed="img/unprocessed/"
		learned ="img/learned/"
		staging ="img/staging/"
		#Move all file to be processed to temp folder
		imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
				
		for imagePath in imagePaths:
			id= int(os.path.split(imagePath)[-1].split('.')[0])
			shutil.move(imagePath,staging+str(id)+".jpg")
 
		# Create index for the images in temp folder
                self.index(staging)  	   
		
		#Find list of files from staging directory
		imagePaths=[os.path.join(staging,f) for f in os.listdir(staging)]
		faces=[]
		ids=[]
		
		if imagePaths != None :
			for imagePath in imagePaths:
				#myfile=path(imagePath)
				if os.path.isfile(imagePath) :
					faceImg=Image.open(imagePath).convert('L');
					faceNP=np.array(faceImg,"uint8")
					oldid= int(os.path.split(imagePath)[-1].split('.')[0])
					ids=self.search(faceNP)
	                        	if ids== None:
						#Move file to unprocessed
						shutil.move(imagePath,unprocessed+str(oldid)+".jpg")

					else:
						UserId=	self.getid()
						self.InsertUSer(UserId,"temp",35,"bangalore")
						for idp in ids:
							imPath=staging + str(idp) + ".jpg"
							procpath=processed+str(UserId)+"." +str(idp)+".jpg"							
							if os.path.isfile(imPath) :
		 						faceImg=Image.open(imPath).convert('L');
								faceNP=np.array(faceImg,"uint8")
                                                		faces.append(faceNP)
								ids.append(UserId)
								#Move file to processed folder	
								shutil.move(imPath,procpath)						 
                                                	
		
	        self.train_model()
		return


	def index(self,Dpath):
		#This procedure is used to index all recorded faces and face features are stored in index.csv
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(self.datapath)
		path=Dpath
		imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
		#print imagePaths
		faces=[]
		ids=[]
		output = open("models/gini/index.csv", "w")

		for imagePath in imagePaths:
			faceImg=cv2.imread(imagePath)
			gray=cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
			gray=self.ImageProcess(gray)
			id= int(os.path.split(imagePath)[-1].split('.')[0])
			print id
			features = self.describenew(gray)

			# write the features to file
			if features != None:
				features = [str(f) for f in features]
				output.write("%s,%s\n" % (id, ",".join(features)))
				cv2.imshow("Indexing", gray)
				cv2.waitKey(20)
	
		output.close()
		cv2.destroyAllWindows
		return

	def ImageProcess(self,gray):

		#Image adjustments for size and contrast
		#Resize image to standard size
		q=130.0/gray.shape[0]			
		r=130.0 / gray.shape[1]
		dim=(int(gray.shape[0]*q),int(gray.shape[1]*r))
		gray=cv2.resize(gray,dim,interpolation=cv2.INTER_AREA)
		#Adjust Contrast and brightness
		gray=cv2.equalizeHist(gray)
		#Remove the rough edges
		kernal=np.ones((5,5),np.float32)/25
		gray=cv2.filter2D(gray,-1,kernal)
		return gray
		
	
	def describenew(self,gray):
		#Extracts unique features for faces . which will the be used to identify the faces
		datapath="models/gini/hogface/shape_predictor_68_face_landmarks.dat"
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(datapath)
		res=[]
		shape=[]
		# Show faces
		cv2.imshow("describe",gray)
		cv2.waitKey(20)	
		gray=np.array(gray,"uint8")
		rects = detector(gray, 1)

		for (i,rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = faceutil.shape_to_np(shape)
			#print shape
		a=len(shape)

		if a > 0:
			for (x,y) in shape:
				res.append(round(math.atan(y/x),6))
			
		else:
			
			res=None
		return res


	def train_model(self):
		#Procedure train the model to recognise faces and corresponding ids
		path="img/processed"
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
		cv2.destroyAllWindows	
		return 

	
	def search(self, image):
		queryFeatures=[]
		queryFeatures= self.describenew(image)
		results = []
		ids=[]
		imgP=None		
                
		if queryFeatures:
			# open the index file for reading
			with open(self.indexPath) as f:	
				reader = csv.reader(f)
				for row in reader:
					features = [float(x) for x in row[1:]]
					#cs =self.sm.cosine_similarity(queryFeatures,features)
					cs =self.sm.euclidean_distance(queryFeatures,features)	
					if cs > 0.95 :
           					print (row[0],cs)
						imageP="img/users/" + row[0] + ".jpg"
						ids.append(row[0])	
			
				f.close()
                else :
			ids=None

		return ids

	def add_User(self,fids):
		cap = cv2.VideoCapture(0)
		samplenum=0
		ids=fids
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
				image = img[y:y+h,x:x+w]
				dst = np.zeros(shape=(5,2))
				norm_image = cv2.normalize(image,dst,0,255,cv2.NORM_MINMAX)
				cv2.imwrite("img/processed/" + str(ids) + "." + str(samplenum) + ".jpg" , norm_image)
				cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
	        		cv2.waitKey(100)
				cv2.imshow("Face",gray)
			cv2.waitKey(1)
			if(samplenum >20):
				break
	 	self.train_model()
		cap.release()
		cv2.destroyAllWindows() 
		return

