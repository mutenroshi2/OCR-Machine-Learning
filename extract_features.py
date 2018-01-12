#import necessary libraries
from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
from img_crop import crop_roi
from hog import HOG
import dataset_alphabets
from conf import Conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
from load_img import img_proc_roi
import cv2

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
data = []
labels = []

# grab the set of positive images
# trnpaths = conf["image_dataset"]
trnpaths = list(paths.list_images(conf["image_dataset"]))
print("[INFO] describing training ROIs...")

#setup the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnpaths),widgets=widgets).start()

#loop over the training paths
for(i,trnpath) in enumerate(trnpaths):
    image = cv2.imread(trnpath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [x, y, w, h] = img_proc_roi(trnpath)
    if (x>0 and y > 0  and w > 0 and h > 0):
        bb = [x,y,w,h]
        # print(trnpath)
        roi = crop_roi(gray,bb,10,dstSize=(32,32))
        # define the list of ROI's that will be described,based on wheather or not the horizontal flip of the image should
        # be used
        rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)
        #loop over the rois
        for roi in rois:
            # extract features from the ROI and update the list of features and labels
            features = hog.describe(roi)
            data.append(features)
            labels.append(1)

            # update the progress bar
            pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()

# grab the distraction image paths and reset the progress bar
dstPaths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROIs...")

# loop over the desired number of distraction images
for i in np.arange(0, conf["num_distraction_images"]):
    # randomly select a distraction image, load it, convert it to grayscale, and
    # then extract random patches from the image
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]),
                                 max_patches=conf["num_distractions_per_image"])

    # loop over the patches
    for patch in patches:
        # extract features from the patch, then update the data and label list
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    # update the progress bar
    pbar.update(i)

# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset_alphabets.dump_dataset(data, labels, conf["features_path"], "features")