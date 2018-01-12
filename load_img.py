# import all the necessary libraries
import cv2
import imutils
from imutils import paths

def img_proc_roi(image):
    # Basic image pre-processing
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernelSizes = [(3, 3), (5, 5), (7, 7)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((5,5), np.uint8)
    # dilate = cv2.dilate(thresh,None)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(opening, ret, ret * 0.95)
    # cv2.imshow("edges",edges)


    ###################################################################################################################
    cnt_area = []
    # finding contours
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # get contours
    contours = contours[0] if imutils.is_cv2() else contours[1]
    [cnt_area.append(cv2.contourArea(contours)) for c in contours]
    if max(cnt_area) > 800:
        [x, y, w, h] = cv2.boundingRect(contours)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        roi = [x,y,w,h]
        return [x,y,w,h]
    else:
        return [0,0,0,0]
    '''
    # Don't plot small false positives that aren't text
    if w < 35 and h < 35:
        continue

    # draw rectangle around contour on copied image
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 255), 2)

    
    #crop image and send to OCR  , false detected will return no text 
    cropped = image[y :y +  h , x : x + w]
    s = '/home/guru/Desktop/handwriting_recognition'+ '/crop_' + str(index) + '.jpg'
    print(s)
    cv2.imwrite(s , cropped)
    index = index + 1
    '''
    # img_small = img[y:h+y, x:w+x]
    # cv2.imshow('captcha_result_1', img)
    # cv2.imshow('captcha_result', img_small)
    # cv2.waitKey(0)
# test = img_proc_roi('test.png')
# print(test)