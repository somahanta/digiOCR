import os
import numpy as np
import cv2
from PIL import Image
import scipy.misc
from scipy.misc import imresize
import matplotlib.pyplot as plt
#from tempfile import TemporaryFile

#outfile = TemporaryFile()

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


def makeBW(dataimg):
	m,n=dataimg.shape
	#convert to pure black and white
	for i in range(0,n):
		for j in range(0,m):
			if dataimg[j][i]<=128:
				dataimg[j][i]=0
			else:
				dataimg[j][i]=255	
	return dataimg


path="/home/soubhik/Desktop/OCR/neural_network/English/Fnt"
# path='.'
#os.chdir(path)
folders= os.listdir (path) # get all files' and folders' names in the current directory


# from pprint import pprint
# pprint(filenames)


myrow=0
dataset=np.zeros([62292,901])
print(dataset)
print(dataset.shape)

for folder in folders: # loop through all the files and folders
    # if os.path.isdir(os.path.join(os.path.abspath("."), filename)): # check whether the current object is a folder or not
    #     result.append(filename)
        print(folder)
        clsval=int(folder[7:])
        newpath=path+"/"+folder
        datafiles= os.listdir (newpath)
        for eachfile in datafiles:
        	# print(eachfile)
            imgpath=path+"/"+folder+"/"+eachfile
            imgfile=cv2.imread(imgpath)
            imgfile=imgfile[:,:,0]
            tmp=trimAndPad(imgfile)
            # plt.imshow(tmp,cmap='gray')
            # plt.show()
        	# imgfile=imresize(imgfile,[30,30])
        	# print(type(imgfile))
        	# print(imgfile.shape)
        	#imgfile=makeBW(imgfile)
        	# plt.imshow(imgfile,cmap='gray')
        	# plt.show()
        	#now add it to dataset array
        	# tmp=np.invert(imgfile)
        	# plt.imshow(tmp,cmap='gray')
        	# plt.show()
            tmp=tmp.ravel()
            #print(tmp.shape)
            tmp=np.append(tmp,clsval)
            tmp = tmp.reshape((1,901))
        	# print(tmp)
        	# print(tmp.shape)

        	#one set of data is here.... pixels followed by the class
        	#dataset=np.concatenate((dataset,tmp),axis=0)
            dataset[myrow:,:]=tmp
            myrow+=1
        	# print(dataset)
        	# print(dataset.shape)
        	
print(dataset)
#finaldataset=dataset[1:,:]
finaldataset=dataset
print('\n\n')
print(finaldataset)
#np.save(outfile,finaldataset)
#Binary data
np.save('dataset1_4.npy', finaldataset)

#Human readable data
#np.savetxt('dataset2_2.txt', finaldataset)