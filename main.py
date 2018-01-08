from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py

img = misc.imread('/home/soubhik/Desktop/OCR/sampletext.jpg')
plt.imshow(img)
plt.show()

arr = np.array(img)
print(arr.size)
print(arr.shape)

#get grayscale image
gray = img[:,:,0]
misc.imsave("sample2.jpg",gray)

#open the grayscale image
img2 = misc.imread('/home/soubhik/Desktop/OCR/sample2.jpg')
plt.imshow(img2)
plt.show()

arr2 = np.array(img2)
print(arr2.size)
print(arr2.shape)