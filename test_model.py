# import the necessary packages
from conf import Conf
from hog import HOG
from objectdetector import ObjectDetector
from nms import non_max_supression
import numpy as np
import imutils
import argparse
import cPickle
import cv2

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c","--conf",required=True,help="Path to the configuration file")
ap.add_argument("-i","--image",required=True,help="Path to the image to be classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

#load the classifier, then initialize the HOG descriptors
# and the object detector
model = cPickle.loads(open(conf["classifier_path"]).read())
hog = HOG(orientations=conf["orientations"],pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]),normalize=conf["normalize"])
od = ObjectDetector(model,hog)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image,width=min(128,image.shape[1]))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
edges = cv2.Canny(opening, ret, ret * 0.95)

# detect objects in the image and apply non-maxima supression to the bounding boxes
(boxes,probs) = od.detect(edges,conf["window_dim"],winstep=conf["window_step"],pyramidscale=conf["pyramid_scale"],
                          minprob=conf["min_probability"])
pick = non_max_supression(np.array(boxes),probs,conf["overlap_thresh"])
orig = image.copy()

# loop over the original bounding boxes and draw them
for (startx,starty,endx,endy) in boxes:
    cv2.rectangle(orig,(startx,starty),(endx,endy),(0,0,255),2)

# loop over the allowed bounding boxes and draw them
for (startx,starty,endx,endy) in pick:
    cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

# show the output images
cv2.imshow("Original",orig)
cv2.imshow("Image",image)
cv2.waitKey(0)