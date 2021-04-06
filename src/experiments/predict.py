import dill
import yaml
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def predict_instance(x, model):
    '''
    Runs model prediction on 1 or more input images.
    :param x: Image(s) to predict
    :param model: A Keras model
    :return: A numpy array comprising a list of class probabilities for each prediction
    '''
    y = model.predict(x)  # Run prediction on the images
    return y


def predict_set(model, preprocessing_func, predict_df):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_func: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    # Create generator to load images from the frames CSV
    img_gen = ImageDataGenerator(preprocessing_function=preprocessing_func)
    img_shape = tuple(cfg['DATA']['IMG_DIM'])
    x_col = 'Frame Path'
    y_col = 'Class Name'
    class_mode = 'categorical'
    generator = img_gen.flow_from_dataframe(dataframe=predict_df, directory=cfg['PATHS']['FRAMES'],
                                            x_col=x_col, y_col=y_col, target_size=img_shape,
                                            batch_size=cfg['TRAIN']['BATCH_SIZE'],
                                            class_mode=class_mode, validate_filenames=True, shuffle=False)
    class_idx_map = dill.load(open(cfg['PATHS']['CLASS_NAME_MAP'], 'rb'))
    class_idx_map = {v: k for k, v in class_idx_map.items()}    # Reverse the map

    # Obtain prediction probabilities
    p = model.predict_generator(generator)
    test_predictions = np.argmax(p, axis=1)

    # Get prediction classes in original labelling system
    pred_classes = [class_idx_map[v] for v in list(test_predictions)]
    test_predictions = [cfg['DATA']['CLASSES'].index(c) for c in pred_classes]
    return test_predictions, p