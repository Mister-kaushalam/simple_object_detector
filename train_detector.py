#importing the necessary libraries
from __future__ import print_function
from imutils import paths
#The annotations/bounding boxes for the CALTECH-101 dataset are actually .mat  files which are Matlab files, therefore we need loadmat
from scipy.io import loadmat
from skimage import io
import argparse
import dlib
import sys

#handle python3 compatibility
if sys.version_info>(3,):
    long=int

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c","--class", required=True, help="Path to caltech-101 images")
ap.add_argument("-a", "--annotations", required=True, help="Path to caltech-101 class annotations")
ap.add_argument("-o","--output", required=True, help="Path to the output detector")
args = vars(ap.parse_args())


# grab the default training options for our HOG + Linear SVM detector, then initialize the
# list of images and bounding boxes used to train the classifier

options =dlib.simple_object_detector_training_options()
images = []
boxes = []

#loop over the image paths and process them
for imagePath in paths.list_images(args['class']):
    #extract the image ID from the image path and load the image annotations file
    imageID = imagePath[imagePath.rfind("/")+1:].split("_")[1]
    imageID=imageID.replace(".jpg","")
    p = "{}/annotation_{}.mat".format(args["annotations"], imageID)
    annotations = loadmat(p)["box_coord"]

    bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h)) for (y,h,x,w) in annotations ]
    boxes.append(bb)

    #add the image to the list of images
    images.append(io.imread(imagePath))

#training the object detector
print("[INFO] training the detector...")
detector = dlib.train_simple_object_detector(images, boxes, options)

#dump the classifier to the file
print("[INFO] dumping the classifier to file...")
detector.save(args["output"])

#visualize the result of the detector
win=dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()




