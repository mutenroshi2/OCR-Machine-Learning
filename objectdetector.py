#import the necessary packages
import helper

class ObjectDetector:
    def __init__(self,model,desc):
        #store the classifier and HOG descriptor
        self.model = model
        self.desc = desc

    def detect(self,image,windim,winstep=2,pyramidscale=1.5,minprob=0.7):
        #initialize the list of bounding boxes and associated probabilities
        boxes=[]
        probs=[]

        #loop over the image pyramid
        for layer in helper.pyramid(image,scale=pyramidscale,minSize=windim):
            # determine the current scale of the pyramid
            scale = image.shape[0]/float(layer.shape[0])

            #loop over the sliding windows for the current pyramid layer
            for(x,y,window) in helper.sliding_window(layer,winstep,windim):
                # grab the dimensions of the window
                (winh,winw) = window.shape[:2]

                #ensure the window dimensions match the supplied sliding window dimensions
                if winh == windim[1] and winw==windim[0]:
                    #extract HOG features from the current window and classify whether or
                    # not this window contains an object we are interested in
                    features = self.desc.describe(window).reshape(1,-1)
                    prob = self.model.predict_proba(features)[0][1]

                    #check to see if the classifier has found an object with sufficient probability
                    if prob> minprob:
                        #compute the (x,y)-coordinates of the bounding box using the current scale of the image pyramid
                        (startx,starty) = (int(scale*x),int(scale*y))
                        endx = int(startx+(scale*winw))
                        endy = int(starty+(scale*winh))

                        #update the list of bounding boxes and probablities
                        boxes.append((startx,starty,endx,endy))
                        probs.append(prob)

        #return a tuple of the bounding boxes and probabilities
        return (boxes,probs)