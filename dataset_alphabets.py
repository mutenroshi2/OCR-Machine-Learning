#import the necessary libraries
import numpy as np
import h5py

def dump_dataset(data,lables,path,datasetname,writemethod='w'):
    #open the database, create the dataset,write the data and lables to dataset and close the database
    db = h5py.File(path,writemethod)
    dataset = db.create_dataset(datasetname,(len(data),len(data[0])+1),dtype="float")
    # print("The length of lables is {}".format(len(lables)))
    # print("The length of data is {}".format(len(data)))
    # print(dataset.shape)
    dataset[0:len(data)] = np.c_[lables,data]
    db.close()

def load_dataset(path,datasetname):
    #open the database, grab the lables and data , close the dataset
    db = h5py.File(path,"r")
    (lables,data) = (db[datasetname][:,0],db[datasetname][:,1:])
    db.close()

    return(data,lables)