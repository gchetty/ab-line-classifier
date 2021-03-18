import pandas as pd
import yaml
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def to_greyscale(img):
    '''
    Converts RBG image to greyscale.
    :param img: RBG image
    :return: Image to convert to greyscale
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def mp4_to_images(mp4_path):
    '''
    Converts masked ultrasound mp4 video to a series of images and saves the images in the same directory.
    :param mp4_path: File name of the mp4 file to convert to series of images.
    '''
    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)      # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]       # Strip file extension

    idx = 0
    max_area = 0
    max_area_id = 0
    while (True):
        ret, frame = vc.read()
        if not ret:
            break   # End of frames reached
        img_path = vid_dir + '/' + mp4_filename + '_' + str(idx) + '.jpg'
        cv2.imwrite(img_path, frame) # Save all the images out
        idx += 1

    '''
    Add sanity check?
    '''
