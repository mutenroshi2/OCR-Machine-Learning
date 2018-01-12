#import the necessary libraries
from __future__ import print_function
import dataset_alphabets
from conf import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import cPickle

#construct the parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c","--conf",required=True,help="Path to the conf file")
ap.add_argument("-n","--hard-negatives",type=int,default=-1,
                help="flag indicating whether or not hard negatives should be used")
args = vars(ap.parse_args())

#load the configuration file and initial dataset
print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data,labels) = dataset_alphabets.load_dataset(conf["features_path"],"features")

#check to see if the hardnegetive flag was supplied
if args["hard_negatives"] > 0:
    print("[INFO] loading hard negatives...")
    (harddata,hardlabels) = dataset_alphabets.load_dataset(conf["features_path"],"hard_negatives")
    data = np.vstack([data,harddata])
    labels = np.hstack([labels,hardlabels])

#train the classifier
print("[INFO] training classifier...")
model = SVC(kernel="linear",C=conf["C"],probability=True,random_state=42)
model.fit(data,labels)

#dump the classifier to file
print("[INFO] dumping the classifier...")
f = open(conf["classifier_path"], "w")
f.write(cPickle.dumps(model))
f.close()