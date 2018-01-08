import numpy as np
import h5py
import cv2
from PIL import Image
import scipy.misc
from scipy.misc import imresize
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

def sigmoid(z):
	tmp2=1.0+np.exp(-z)
	g=1.0 / tmp2
	return g 

def sigmoidGrad(X):
	# * implies element wise multiplication
	y = sigmoid(X)
	return y * (1 - y)

def predict(theta1, theta2, X):
	m,n=X.shape

	a1=X
	a1=np.hstack((np.ones((m,1)),a1))
	z2=a1.dot(theta1.T)
	a2=sigmoid(z2)

	m2,n2=a2.shape
	a2=np.hstack((np.ones((m2,1)),a2))
	hx=a2.dot(theta2.T)
	hx=sigmoid(hx)

	pr=np.argmax(hx,axis=1)

	return pr.T 



def cost(params, X, y, Lambda, inputlayer, numlabels, hiddenlayer):
	
	theta1= np.array(np.reshape(params[:hiddenlayer*(inputlayer+1)],(hiddenlayer,inputlayer+1)))
	theta2= np.array(np.reshape(params[hiddenlayer*(inputlayer+1):],(numlabels,hiddenlayer+1)))
	
	#calculate cost / FWD PROP

	J=0
	m,n=X.shape
	theta1_grad=np.zeros(theta1.shape)
	theta2_grad=np.zeros(theta2.shape)

	a1=X
	a1=np.hstack((np.ones((m,1)),a1))
	z2=a1.dot(theta1.T)
	a2=sigmoid(z2)

	m2,n2=a2.shape
	a2=np.hstack((np.ones((m2,1)),a2))
	hx=a2.dot(theta2.T)
	hx=sigmoid(hx)

	# print(hx)
	# print(hx.shape)
	
	#cost J(theta)=(1/m) sum(1to m)sum(1 to numlabels)(-y log hx - (1-y) log(1-hx))+Lambda/2m(squared sum of all weights)

	# print(y.shape)
	# print(hx.shape)

	J = -y * np.log(hx) -(1 - y) * np.log(1 - hx)
	J=J.sum()/m
	# print(J.shape)

	theta2[:,0] = 0
	theta1[:,0] = 0
	
	theta1sq = theta1 ** 2
	theta2sq = theta2 ** 2

	sum = theta1sq.sum() + theta2sq.sum();
	J = J + sum * Lambda * 1.0 /(2 * m)




	#calculate gradient / BACKPROPAGATION

	d3=hx-y

	theta22 = theta2[:, 1:] 
	d2 = (theta22.T.dot(d3.T)) * sigmoidGrad(z2).T;
	# print('d2 shape is: ')
	# print(d2.shape)

	delta1 = d2.dot(a1)
	delta2 = d3.T.dot(a2)

	delta1 = delta1 / m * 1.0  #make float
	delta2 = delta2 / m * 1.0

	theta1_grad = delta1 + theta1 * Lambda / m * 1.0
	theta2_grad = delta2 + theta2 * Lambda / m * 1.0

	grad=np.hstack((theta1_grad.ravel(),theta2_grad.ravel()));

	return (J,grad)


#load training data
def loadData():
	print('loading data !')
	dataset=np.load('dataset1_4.npy')
	print(dataset.shape)
	#print(n1)
	np.random.shuffle(dataset)	

	m,n=dataset.shape
	Y=dataset[:,n-1]
	Y=Y.transpose()
	Y=Y.reshape((m,1))
	dataset=dataset[:, 0:n-1]
	dataset=dataset.astype(float)
	dataset=dataset/255

	onehot_encoder = OneHotEncoder(sparse=False)
	print(Y.shape)
	Y = onehot_encoder.fit_transform(Y)
	#print(dataset)

	#now split into training data, validation data and test data
	xtrain, xtest, ytrain, ytest = train_test_split(dataset, Y, test_size=0.4)
	xval, xtest, yval, ytest = train_test_split(xtest, ytest, test_size=0.5)

	print('shape of xtrain, ytrain, xval, yval, xtest, ytest: ');
	print(xtrain.shape)
	print(ytrain.shape)
	print(xval.shape)
	print(yval.shape)
	print(xtest.shape)
	print(ytest.shape)

	return (xtrain, ytrain, xval, yval, xtest, ytest)

def run():
	xtrain, ytrain, xval, yval, xtest, ytest=loadData()

	m,n=xtrain.shape

	inputlayer=n
	hiddenlayer=100
	numlabels=62

	print('\nInput layer size = '+str(inputlayer))

	theta1=np.random.uniform(low=-0.25, high=0.25, size=(hiddenlayer,inputlayer+1))   #-0.5 to +0.5
	theta2=np.random.uniform(low=-0.25, high=0.25, size=(numlabels,hiddenlayer+1))

	print(theta1.shape)
	print(theta2.shape)

	params=np.hstack((theta1.ravel(),theta2.ravel())); #this is horizontal, we need vertical

	print('params shape: ')
	#m=(hiddenlayer*(inputlayer+1)+numlabels*(hiddenlayer+1))
	m=params.shape[0]
	params=params.reshape((m,1))
	print(params.shape)
	theta=params

	batch_size = 15000  #train using the first 10000 examples only, not all 40000
	rang = 10 #repeat num
	

	Lambda = input("Enter lambda : ")
	num_iter = input("Enter number of iterations for gradient : ")
	
	for i in range(rang):
		print('\n\nBatch no '+str(i+1))

		# combine, shuffle, and then use			
		# allbatch=np.hstack((X_batch,Y_batch))
		# np.random.shuffle(allbatch)
		# X_batch=allbatch[:,:n]
		# Y_batch=allbatch[:,n]
		# Y_batch=Y_batch.reshape((Y_batch.shape[0],1))

		X_batch = xtrain[:batch_size ,:]
		Y_batch = ytrain[:batch_size ,:]

		J,grad= cost(theta, X_batch, Y_batch, Lambda, inputlayer, numlabels, hiddenlayer)

		print('Cost='+str(J))

		ybatch_val=np.argmax(Y_batch,axis=1)
		ybatch_val=ybatch_val.T

		predval=predict(theta1,theta2,X_batch)
		chkval = (predval == ybatch_val) * 1
		acc =  1.0 * sum(chkval)/len(chkval)
		print "Accuracy in batch is : %f" %(acc*100)

		tmp1=np.argmax(yval,axis=1)
		predval=predict(theta1,theta2,xval)
		chkval = (predval == tmp1.T) * 1
		# print(sum(chkval))
		# print(len(chkval))
		acc =  1.0 * sum(chkval)/len(chkval)
		print "Accuracy in validation set is : %f" %(acc*100)

		tmp2=np.argmax(ytest,axis=1)
		predval=predict(theta1,theta2,xtest)
		chkval = (predval == tmp2.T) * 1
		acc =  1.0 * sum(chkval)/len(chkval)
		print "Accuracy in test set is : %f" %(acc*100)

		#minimize cost function

		Result = op.minimize(fun = cost, x0 = theta, args = (X_batch,Y_batch,Lambda,inputlayer,numlabels,hiddenlayer), method = 'TNC',
			jac = True, options = {'maxiter': num_iter})
		theta = Result.x;
		#updated NEW THETA

	
		theta1= np.array(np.reshape(theta[:hiddenlayer*(inputlayer+1)],(hiddenlayer,inputlayer+1)))
		theta2= np.array(np.reshape(theta[hiddenlayer*(inputlayer+1):],(numlabels,hiddenlayer+1)))
	
		pred = predict(theta1, theta2, X_batch)
		pred = pred.T
		corr = (pred == ybatch_val) * 1;

		acc =  1.0 *sum(corr)/len(corr)
		print "Accuracy in batch after descent is : %f \n" %(acc*100)

	print('Training complete')

	predval=predict(theta1,theta2,xval)
	chkval = (predval == tmp1.T) * 1
	# print(sum(chkval))
	# print(len(chkval))
	acc =  1.0 * sum(chkval)/len(chkval)
	print "Accuracy in validation set is : %f" %(acc*100)

	tmp2=np.argmax(ytest,axis=1)
	predval=predict(theta1,theta2,xtest)
	chkval = (predval == tmp2.T) * 1
	acc =  1.0 * sum(chkval)/len(chkval)
	print "Accuracy in test set is : %f" %(acc*100)


	np.save('theta1.npy', theta1)
	np.save('theta2.npy', theta2)
	return

if __name__=="__main__":
	print('hello there')
	run()