# OCR-Machine-Learning

The training and the testing data can be downloaded from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/


This code uses 6 step framework method learnt from the coursework and certification https://www.pyimagesearch.com/pyimagesearch-gurus/

Sample P positive samples from the dataset and HOG is extracted from these samples.


Negative N samples from the dataset are taken HOG is extracted from these samples as well.


Machine is trained using Linear SVM on both positive and negative samples.


Hard negative mining is performed.


Machine is trained using Linear SVM on both positive,negative and hard-negative samples.


Image pyramid and sliding windows are applied to identify the required object.
