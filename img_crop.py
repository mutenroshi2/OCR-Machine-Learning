# import the necessary packages
import imutils
import cv2


def crop_roi(image, bb, padding=10, dstSize=(32,32)):
    # unpack the bounding box and extract the ROI from the image, while taking into account
    # the supplied offset
    # print(image.shape)
    (x, y, w, h) = bb
    # (x, y) = (max(x - padding, 0), max(y - padding, 0))
    roi = image[y:h+y, x:w+x]

    # resize the ROI to the desired destination size
    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)
    # print(roi.shape)
    # cv2.imshow("roi_test",roi)
    # cv2.waitKey(0)

    # return the ROI
    return roi