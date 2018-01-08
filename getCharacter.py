#file to extract each character from a word, PAD it to form a square
import numpy as np
import cv2
import os
from PIL import Image
import scipy.misc
from scipy.misc import imresize
import matplotlib.pyplot as plt


def sigmoid(z):
	tmp2=1.0+np.exp(-z)
	g=1.0 / tmp2
	return g 

def predictChar(theta1, theta2, X):
	
	X=X.ravel()
	X=X.reshape((1,X.shape[0]))
	#print(X.shape)
	m,n=X.shape
	X=(X * 1.0)/255.0

	a1=X
	a1=np.hstack((np.ones((m,1)),a1))
	z2=a1.dot(theta1.T)
	a2=sigmoid(z2)

	m2,n2=a2.shape
	a2=np.hstack((np.ones((m2,1)),a2))
	hx=a2.dot(theta2.T)
	hx=sigmoid(hx)

	pr=np.argmax(hx,axis=1)

	#print(pr)
	return (pr[0]+1)

def convertBW(convimg):
	m,n=convimg.shape

	#convert to pure black and white
	for i in range(0,n):
		for j in range(0,m):
			if convimg[j][i]<=128:
				convimg[j][i]=0
			else:
				convimg[j][i]=255	
	return convimg



def trimAndPad(letter):
	myletter=letter
	myx,myy=myletter.shape
	#trim whites from all sides
	f=0 
	#top
	s1=0
	s2=0
	s3=0
	s4=0
	for i in range(0,myx):
		for j in range(0,myy):
			if myletter[i][j]==0:
				s1=i
				f=1
				break;
		if f==1:
			break

	f=0 
	#bottom
	for i in range(myx-1,0,-1):
		for j in range(0,myy):
			if myletter[i][j]==0:
				s2=i
				f=1
				break;
		if f==1:
			break

	f=0 
	#left
	for j in range(0,myy):
		for i in range(0,myx):
			if myletter[i][j]==0:
				s3=j
				f=1
				break;
		if f==1:
			break

	f=0 
	#right
	for j in range(myy-1,0,-1):
		for i in range(0,myx):
			if myletter[i][j]==0:
				s4=j
				f=1
				break;
		if f==1:
			break

	myletter=myletter[s1:s2, s3:s4]

	#now pad and make SQUARE
	myx,myy=myletter.shape
	if myx>myy: #height is more, pad left right
		diff=myx-myy
		half=diff/2
		sqletter = np.zeros(shape=(myx,myx))
		for i in range(0,myx):
			for j in range(0,myx):
				sqletter[i][j]=255

		for i in range(0,myx):
			for j in range(half,half+myy):
				sqletter[i][j]=myletter[i][j-half]
	else:
		diff=myy-myx
		half=diff/2
		sqletter = np.zeros(shape=(myy,myy))
		for i in range(0,myy):
			for j in range(0,myy):
				sqletter[i][j]=255

		for i in range(half,myx+half):
			for j in range(0,myy):
				sqletter[i][j]=myletter[i-half][j]

	myletter=sqletter
	myletter=imresize(myletter,[30,30])
	myletter=np.invert(myletter)
	#myletter=convertBW(myletter)
	# plt.imshow(sqletter, cmap='gray')
	# plt.show()
	return myletter




def run(dict, theta1, theta2, imgfile):

	#imgfile = cv2.imread("myword.jpg")
	# cv2.imshow("The image", imgfile)
	# cv2.waitKey(0)

	# print(type(imgfile))
	# print(imgfile.shape)

	#convert from 3d to 2d
	imgfile=imgfile[:,:,0]
	# print(imgfile.shape)



	#display after B&W full once
	# cv2.imshow("The Black and White image", imgfile)
	# cv2.waitKey(0)

	# plt.imshow(imgfile, cmap='gray')
	# plt.show()

	print("Now we need to divide letters in this word")
	matx, maty=imgfile.shape	
	#print('The shape of this word is : '+str(matx)+' by '+str(maty))

	#trim each word from right and left (ALL WHITE SPACES)

	#for proper trim, convert to B&W first

	#imgfile2=convertBW(imgfile)

	#from right
	flag11=0
	rmargin=0
	for y in range(maty-1,0,-1):
		for x in range(0,matx):
			if imgfile[x][y]==0:
				flag11=1;
				rmargin=y
				break
		if flag11==1:
			break;

	#from left
	#from right
	flag11=0
	lmargin=0
	for y in range(0,maty):
		for x in range(0,matx):
			if imgfile[x][y]==0:
				flag11=1;
				lmargin=y
				break
		if flag11==1:
			break;

	# print('lmargin rmargin are: ')
	# print(lmargin)
	# print(rmargin)

	imgfile=imgfile[:,lmargin:rmargin]

	matx, maty=imgfile.shape

	plt.imshow(imgfile, cmap='gray')
	plt.show()

	#print('The shape of this word is : '+str(matx)+' by '+str(maty))





	threshold= 2  #letters are separated by at least 2 pixels
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
				if imgfile[j][i]==0:
					flag=0
					break
			if flag==0:
				break
		if flag==1:
			wordend=i2
		#	print('letter from '+str(wordbg)+' to '+str(wordend))
			thisword=imgfile[0:matx,wordbg:wordend]
			thisword=trimAndPad(thisword)
			# plt.imshow(thisword, cmap='gray')
			# plt.show()
			letterval=predictChar(theta1,theta2,thisword)
			print dict[letterval]
			i2=i2+threshold
			#start i2 again from next word
			while True:
				flag2=1
				for i in range(0,matx):
					if imgfile[i][i2]==0:
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
			#print('the last letter of this line is from '+str(wordbg)+' to '+str(maty))
			lastword=imgfile[0:matx,wordbg:maty]
			lastword=trimAndPad(lastword)
			# plt.imshow(lastword, cmap='gray')
			# plt.show()
			letterval=predictChar(theta1,theta2,lastword)
			print dict[letterval]
			# scipy.misc.imsave('myword.jpg', lastword)
			break;
	return

if __name__=="__main__":
	print('trying to get characters')
	dict = {1: '0', 2: '1', 3: '2', 4: '3', 5:'4',6:'5',7:'6',8:'7',9:'8',10:'9',
		    11:'A',12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',18:'H',19:'I',20:'J',
		    21:'K',22:'L',23:'M',24:'N',25:'O',26:'P',27:'Q',28:'R',29:'S',30:'T',
			31:'U',32:'V',33:'W',34:'X',35:'Y',36:'Z',37:'a',38:'b',39:'c',40:'d',
			41:'e',42:'f',43:'g',44:'h',45:'i',46:'j',47:'k',48:'l',49:'m',50:'n',
			51:'o',52:'p',53:'q',54:'r',55:'s',56:'t',57:'u',58:'v',59:'w',60:'x',
			61:'y',62:'z'}

	theta1=np.load('neural_network/theta1.npy')
	theta2=np.load('neural_network/theta2.npy')

	print(theta1.shape)
	print(theta2.shape)

	path="/home/soubhik/Desktop/OCR/allwords"
	files= os.listdir (path)
	#files.sort()
	print(files)

	for myfile in files:
		# print('checkLoop')
		imgpath=path+"/"+myfile
		# print('chk2')
		imgfile=cv2.imread(imgpath)
		# print('chk3')
		run(dict, theta1, theta2,  imgfile)
		# print('\nnext')