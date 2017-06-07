# import the necessary packages

import numpy as np
import argparse
import imutils
import dlib
import cv2


def rect_to_bb(rect):
	x=rect.left()
	y=rect.top()
	w=rect.right()-x
	h=rect.bottom()-y
	return (x,y,w,h)

def shape_to_np(shape,dtype="int"):
	coords=np.zeros((68,2), dtype=dtype)
	for i in range(0,68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords



	
