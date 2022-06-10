import os.path

import cv2
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_50v2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import matplotlib.pyplot as plt
import pandas as pd
import onnx
from onnx_tf.backend import prepare

from tensorflow.keras.models import load_model

def AB_classifier_preprocess(image, preprocessing_fn):
    '''
    Given a masked ultrasound image, execute preprocessing steps specific to the AB classifier. Specifically, the image
    is resized to (128, 128) and zero-centered with respect to the ImageNet dataset. The result is an image that is
    ready for the forward pass of the view classifier.
    :image (np.array): A masked image with shape (1, H, W, 3)
    :return (np.array): Preprocessed image with shape (1, 128, 128, 3)
    '''

    N_CHANNELS = 3
    INPUT_SIZE = (128, 128)

    # Resize image
    resized_image = cv2.resize(image[0], INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.reshape((1, INPUT_SIZE[0], INPUT_SIZE[1], N_CHANNELS))

    # Apply scaling function
    preprocessed_image = preprocessing_fn(resized_image)
    return preprocessed_image

def view_classifier_preprocess(image):
    '''
    Given a masked ultrasound image, execute preprocessing steps specific to the view classifier. Specifically, the
    image is resized to (128, 128) and zero-centered with respect to the ImageNet dataset. The result is an image that
    is ready for the forward pass of the COVID-19 classifier.
    :image (np.array): A masked image with shape (1, H, W, 3)
    :return (np.array): Preprocessed image with shape (1, 128, 128, 3)
    '''

    N_CHANNELS = 3
    INPUT_SIZE = (128, 128)

    # Resize image
    resized_image = cv2.resize(image[0], INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.reshape((1, INPUT_SIZE[0], INPUT_SIZE[1], N_CHANNELS))

    # Apply scaling function
    preprocessed_image = resnet_50v2_preprocess(resized_image)
    return preprocessed_image

def covid_classifier_preprocess(image):
    '''
    Given a masked ultrasound image, execute preprocessing steps specific to the COVID-19 classifier. Specifically, the image
    is resized to (600, 600) and scaled to [-1, 1]. Then it is 0-centered and subsequently divided by its standard
    deviation.
    :image (np.array): A masked image with shape (1, H, W, 3)
    :return (np.array): Preprocessed image with shape (1, 128, 128, 3)
    '''

    N_CHANNELS = 3
    INPUT_SIZE = (600, 600)

    # Resize image
    resized_image = cv2.resize(image[0], INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.reshape((1, INPUT_SIZE[0], INPUT_SIZE[1], N_CHANNELS))

    # Apply scaling function
    preprocessed_image = xception_preprocess(resized_image)

    # As in keras ImageDataGenerator, set samplewise mean to 0 and divide by standard deviation
    preprocessed_image -= np.mean(preprocessed_image, keepdims=True)
    preprocessed_image /= (np.std(preprocessed_image, keepdims=True) + 1e-6)
    return preprocessed_image

def predict_wavebase_mp4(model_path, mp4_path, preds_path):

    model_ext = os.path.splitext(model_path)[1]
    vc = cv2.VideoCapture(mp4_path)
    if model_ext == '.onnx':
        model = prepare(onnx.load(model_path))
    else:
        model = load_model(model_path, compile=False)
    preds = []
    while (True):
        ret, frame = vc.read()
        if not ret:
            break
        #frame[0:frame.shape[0]//8,:] = 0.
        frame = np.expand_dims(frame, axis=0)
        preprocessed_frame = AB_classifier_preprocess(frame, vgg16_preprocess)
        if model_ext == '.onnx':
            pred = model.run(preprocessed_frame).output
        else:
            pred = model(preprocessed_frame)
        preds += [pred]
    preds = np.vstack(preds)
    pred_df = pd.DataFrame({'Frame': np.arange(preds.shape[0]), 'A lines': preds[:,0], 'B lines': preds[:,1]})
    pred_df.to_csv(preds_path, index=False)
    return preds

#model_path = 'results/models/cutoffvgg16_final_cropped.h5'
model_path = 'results/models/ab_model_not_compiled/AB_classifier.onnx'
mp4_path = 'C:/Users/Blake/Downloads/AB_test/demo.mp4'
preds_path = 'C:/Users/Blake/Downloads/AB_test/demo.csv'
preds = predict_wavebase_mp4(model_path, mp4_path, preds_path)

