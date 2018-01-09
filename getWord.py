#file to extract each line from the final cropped image and extract words from it
import os 
import numpy as np
import cv2
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import getCharacter

def getBW(imgfile):
	#convert to pure black and white
	m,n=imgfile.shape
	for i in range(0,n):
		for j in range(0,m):
			if imgfile[j][i]<=128:
				imgfile[j][i]=0
			else:
				imgfile[j][i]=255	

	#display after B&W full once
	# cv2.imshow("The Black and White image", imgfile)
	# cv2.waitKey(0)

	plt.imshow(imgfile, cmap='gray')
	plt.show()
	return imgfile


def getWordImg():
	imgfile = cv2.imread("bwsample.jpg")
	# cv2.imshow("The image", imgfile)
	# cv2.waitKey(0)

	print(type(imgfile))
	print(imgfile.shape)

	#convert from 3d to 2d
	imgfile=imgfile[:,:,0]
	print(imgfile.shape)

	m,n=imgfile.shape

	#create a black and white copy of image file
	imgfile=getBW(imgfile)

	#get each line
	#search for first black point, save as start, continue till not all white, if no strt found, its over
	#0 is black
	linei=0
	mywordval=0
	while True:
		strt=-1
		lst=-1
		while linei!=m:
			for linej in range(0,n):
				if imgfile[linei][linej]==0:
					strt=linei;
					break;
			if strt!=-1:
				break
			linei=linei+1
		if linei==m:   #or strt!=-1
			break;
		#now find end of line
		while linei!=m:
			flag=1
			for j in range(0,n):
				if imgfile[linei][j]==0:
					flag=0;
					break;
			if flag==1: # all white means end of line
				lst=linei
				break
			linei=linei+1
		# print('A line from'+str(strt)+' to '+str(lst))

		#got the lines, now get the words

		thisline=imgfile[strt:lst, 0:n]
		# cv2.imshow("One line",thisline)
		# cv2.waitKey(0)
		matx, maty=thisline.shape	

		#trim the right side of white space from each line

		#from right
		flag11=0
		rmargin=0
		for y in range(maty-1,0,-1):
			for x in range(0,matx):
				if thisline[x][y]==0:
					flag11=1;
					rmargin=y
					break
			if flag11==1:
				break;

		#from left
		flag11=0
		lmargin=0
		for y in range(0,maty):
			for x in range(0,matx):
				if thisline[x][y]==0:
					flag11=1;
					lmargin=y
					break
			if flag11==1:
				break;

		thisline=thisline[:,lmargin:rmargin]

		




		# print("Now we need to divide words in this line")
		matx, maty=thisline.shape	
		# print('The shape of this line is : '+str(matx)+' by '+str(maty))

		plt.imshow(thisline, cmap='gray')
		plt.show()


		threshold= 35  #words are separated by at least 35 pixels
		#bg, ed
		wordbg=0
		i2=0
		#take the first line to work with
		# myline=imgfile[strt:lst, 0:maty]
		# linex, liney = myline.shape
		# print('The shape of MY line is : '+str(linex)+' by '+str(liney))	
		# while True:
		thelast=maty-threshold-1
		while True:
			if i2==thelast:
				break;
			#print('start i2='+str(i2))
			#check threshold square/rectangle for all white,if yes, the it is separator / SLIDE a rectangle to find spaces
			flag=1
			for i in range(i2,i2+threshold):
				for j in range(0,matx):
					if thisline[j][i]==0:
						flag=0
						break
				if flag==0:
					break
			if flag==1:
				wordend=i2
				#print('word from '+str(wordbg)+' to '+str(wordend))
				thisword=thisline[0:matx,wordbg:wordend]
				plt.imshow(thisword, cmap='gray')
				plt.show()


				# scipy.misc.imsave('/home/soubhik/Desktop/OCR/allwords/myword'+str(mywordval)+'.jpg', thisword)
				getCharacter.run(thisword)
				#os.system('python getCharacter.py')


				i2=i2+threshold
				#start i2 again from next word
				while True:
					flag2=1
					for i in range(0,matx):
						if thisline[i][i2]==0:
							flag2=0
							break
					if flag2==0:
						break;
					i2+=1
				wordbg=i2
			# i2=i2+1
			# if i2==maty:
			# 	break #this line complete
			#print('last i2='+str(i2))
			i2+=1
			if i2==thelast:
				#print('the last word of this line is ')
				lastword=thisline[0:matx,wordbg:maty]
				plt.imshow(lastword, cmap='gray')
				plt.show()
				# scipy.misc.imsave('/home/soubhik/Desktop/OCR/allwords/myword'+str(mywordval)+'.jpg', lastword)
				getCharacter.run(lastword)
				#os.system('python getCharacter.py')
				break;

			mywordval+=1

		print('\n')
		# print('one line of words complete, next line\n\n')


if __name__ == "__main__":
	getWordImg();
