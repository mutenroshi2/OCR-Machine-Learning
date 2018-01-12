#import the required libraries
import imutils

def pyramid(image,scale=2,minSize=(30,30)):
	#yield the original image
	yield image

	#keep looping over the pyramid
	while True:
		#compute the new dimensions of the image and resize it
		w = int(image.shape[1]/scale)
		image = imutils.resize(image,width=w)
		
		#if the resized image is lower than the minSize then 			#break the loop
		if (image.shape[0] < minSize[1] or image.shape[1]<minSize[0]):
			break
		
		#yield the image in the pyramid
		yield image

def sliding_window(image,stepsize,window_size):
	#slide a window across an image
	for y in xrange(0,image.shape[1],stepsize):
		for x in xrange(0,image.shape[0],stepsize):
			#yield the current window
			yield(x,y,image[y:y+window_size[1],x:x+window_size[0]])