# import the necessary libraries
import numpy as np

def non_max_supression(boxes,probs,overlapthresh):
    #if there are no boxes, return an empty list
    #print("The boxes are {}".format(boxes))
    if len(boxes) == 0:
        return []

    #if the bounding boxes are integers convert them to floats - important since we will be doing some division
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype("float")

    #initialize the list of picked indexes
    pick = []

    #grab the co-ordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #compute the area of the bounding boxes and sort the bounding boxes by their associated probabilities
    area = (x2-x1 + 1)*(y2-y1+1)
    idxs = np.argsort(probs)

    # keep looping while some some indexes still remain in the indexes list
    while len(idxs) > 0:
        #grab the last index in the indexes list and add the index value to the list of the
        # picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x,y) co-ordinates for the start of the bounding box and the
        # smallest (x,y) co-ordinates for the end of the bounding box
        xx1 = np.maximum(x1[i],x1[idxs[:last]])
        yy1 = np.maximum(y1[i],y1[idxs[:last]])
        xx2 = np.minimum(x2[i],x2[idxs[:last]])
        yy2 = np.minimum(y2[i],y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)

        #compute the ratio of the overlap
        overlap = (w * h)/area[idxs[:last]]

        #delete all the indexes from the index list that have overlap greater then the
        # provided threshold
        idxs = np.delete(idxs,np.concatenate(([last],np.where(overlap > overlapthresh)[0])))

    # return the bounding boxes that were picked
    return boxes[pick].astype("int")