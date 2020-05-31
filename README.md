# Simple Object Detector using dlib

This is repository containing code of simple object detection method using dlib

### Objective:

> - Leveraging the dlib library to train a classifier in detecting the presence of stop signs in images.

### The DataSet:

Introduced by Fei-Fei et al. in 2004, the [CALTECH-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) dataset is a very popular benchmark dataset for object detection and has been used by many researchers, academics, and computer vision developers to evaluate their object detection algorithms.

The dataset includes 101 categories, spanning a diverse range of objects including elephants, bicycles, soccer balls, and even human brains, just to name a few.

We'll be using the stop_sign class of the dataset to train and locate stop signs in an image. **If you are downloading it directly then you'll also have to download the annotation file for the images** . These are the bounding boxes of objects in images which will be necessary for this code. I have already give the annotation file and images for stop signs in this repository.

For each image in the dataset, an associated bounding box (i.e. (x, y)-coordinates of the object) is provided. Our goal is to take both the images and the bounding boxes (i.e. the annotations) and train a classifier to detect the presence of a given object in an image.

### Prerequisites

1. First you have to install dlib
```python
pip install dlib
```

2. (OPTIONAL) Install cv2. Since I have only used opencv to draw rectange and show images you can substitue it with any library you wish. 
There is an amazing guide to install opencv on your mac, RaspberryPi or ubuntu machine by Adrian Rosebrook, [Install opencv](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/).


3. Scipy, imutils and skimage required
```python
pip install scipy
pip install imutils
pip install skimage
```


## Runing the Code

1. First, you have to train your model by running train_detector.py
```
$ python train_detector.py --class stop_sign --annotations stop_sign_annotations \
	--output output/stop_sign_detector.svm
```
*if you have to full CALTECH-101 dataset, you can replace the class argument and annotation argument to train on a different object class instead of stop signs. Make sure to change the output file name as well.

**it would look something like this**

![Img](/results/dumping_classifier.png)

2. The second step is to test out detector by running test_detector.py. **I have uploaded few test images for stop signs, if you are training on a different object class you have to collect test images yourself **

```
python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing
```
## Results

![Img](/results/ex1.png)


![Img](/results/ex2.png)


![Img](/results/ex3.png)

### Author
* Kaushal Bhavsar


### Acknowledgments

This project is inspired by work of Adrian Rosebrook, Owner, author at PyImageSearch.com. 
