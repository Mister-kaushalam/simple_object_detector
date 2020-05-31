#importing the necessary packages
from imutils import paths
import argparse
import dlib
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to the trained detector")
ap.add_argument("-t","--testing", required=True, help="Path to directory of testing images")
args=vars(ap.parse_args())

#load the detector
detector = dlib.simple_object_detector(args["detector"])

#loop over the testing images
for imagePath in paths.list_files(args["testing"]):
    #load the images and make predictions
    images = cv2.imread(imagePath)
    boxes = detector(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

    #loop over the bounding boxes and draw them over the images
    for b in boxes:
        (x,y,w,h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.rectangle(images, (x,y), (w,h), (0,255,0), 2)

    #show the image
    cv2.imshow("Image", images)
    cv2.waitKey(0)

