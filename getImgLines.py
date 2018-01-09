# import the necessary packages
import numpy as np
import cv2
from PIL import Image
import scipy.misc
import getWord

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
 
#increase contrast

#(THINK WHETHER REQUIRED OR NOT FOR BETTER OCR)
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def run():
	# load the image, convert it to grayscale, and blur it slightly
	image = cv2.imread("paraSample.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	auto = auto_canny(blurred)

	# show the images
	'''cv2.imshow("Original", image)
	cv2.imshow("Canny", auto)
	cv2.waitKey(0)'''


	#perform dilation on the image (to join all white spaces and form regions of text)
	kernel = np.ones((8,8), np.uint8)  #can change value of 10 later
	 
	# The first parameter is the original image,
	# kernel is the matrix with which image is 
	# convolved and third parameter is the number 
	# of iterations, which will determine how much 
	# you want to erode/dilate a given image. 
	img_dilation = cv2.dilate(auto, kernel, iterations=1)
	 
	#cv2.imshow('Input', auto)
	'''cv2.imshow('Dilation', img_dilation)
	 
	cv2.waitKey(0)'''


	#detect region of text and crop it
	# crop the middle portion of text by first detecting the leftmost, rightmost, topmost and bottom most while points 
	# and removing the area to its left /right / bottom / topmost

	img2= img_dilation
	print('dimensions of the image is : ',img2.shape)

	#the image is a numpy array
	m , n = img2.shape
	print(m)
	print(n)

	'''for i in range(0,m):
		for j in range(0,n):
			print(img2[i][j]),
		print('\n')'''

	# 0 is for black, 255 is white
	# save the top, left, right, bottom most white points

	#topmost
	flg=0

	for i in range(0,m):
		for j in range(0,n):
			if img2[i][j]==255:
				top=i
				flg=1
				break
		if flg==1:
			break		

	print('topmost row with white = '+str(top))

	#bottommost , start checking from bottom so that middle black lines not skipped
	flg=0
	for i in range(m-1,top,-1):
		for j in range(0,n):
			if img2[i][j]==255:
				bot=i
				flg=1
				break
		if flg==1:
			break;

	print('bottommost row with white = '+str(bot))

	#leftmost
	flg=0
	for j in range(0,n):
		for i in range(0,m):
			if img2[i][j]==255:
				lft=j
				flg=1
				break
		if flg==1:
			break		
	print('leftmost row with white = '+str(lft))

	#rightmost
	flg=0
	for j in range(n-1,lft,-1):
		for i in range(0,m):
			if img2[i][j]==255:
				flg=1
				ryt=j;
				break
		if flg==1:
			break;

	print('rightmost row with white = '+str(ryt))

	#CROP the image
	crop_img = image[top:bot, lft:ryt]
	'''cv2.imshow("cropped", crop_img)
	cv2.waitKey(0)'''

	# scipy.misc.imsave('outfile.jpg', crop_img)
	scipy.misc.imsave('bwsample.jpg', crop_img)

	# finalimg=change_contrast(Image.open('outfile.jpg'), 100)
	# finalimg=change_contrast(crop_img, 100)
	# finalimg.save('bwsample2.jpg')


	#make into black and white
	# bwimg=Image.open('bwsample2.jpg')
	# bwimg = bwimg.convert('1') 
	# finalimg.save('bwsample.jpg')

if __name__=="__main__":
	run()
	getWord.getWordImg()
