import dill
import yaml
import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess
from src.visualization.visualization import *

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

def get_preprocessing_function(model_type):
    '''
    Get the preprocessing function according to the type of TensorFlow model
    :param model_type: The pretrained model
    :return: A reference to the appropriate preprocessing function
    '''
    if model_type == 'custom_resnetv2':
        return resnet_preprocess
    elif model_type == 'vgg16':
        return vgg16_preprocess
    elif model_type == 'mobilenetv2':
        return mobilenetv2_preprocess
    elif model_type == 'inceptionresnetv2':
        return inceptionresnetv2_preprocess
    elif model_type == 'cutoffvgg16':
        return xception_preprocess
    else:
        return None

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

def compute_metrics(cfg, labels, preds, probs=None):
    '''
    Given labels and predictions, compute some common performance metrics
    :param cfg: project config
    :param labels: List of labels
    :param preds: List of predicted classes
    :param probs: Array of predicted classwise probabilities
    :return: A dictionary of metrics
    '''

    metrics = {}
    class_names = cfg['DATA']['CLASSES']

    precisions = precision_score(labels, preds, average=None)
    recalls = recall_score(labels, preds, average=None)
    f1s = f1_score(labels, preds, average=None)

    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['precision'] = {class_names[i]:precisions[i] for i in range(len(precisions))}
    metrics['recall'] = {class_names[i]:recalls[i] for i in range(len(recalls))}
    metrics['f1'] = {class_names[i]:f1s[i] for i in range(len(f1s))}
    metrics['accuracy'] = accuracy_score(labels, preds)

    if probs is not None:
        metrics['macro_mean_auc'] = roc_auc_score(labels, probs[:,1], average='macro', multi_class='ovr')
        metrics['weighted_mean_auc'] = roc_auc_score(labels, probs[:,1], average='weighted', multi_class='ovr')

        # Calculate classwise AUCs
        for class_name in class_names:
            classwise_labels = (labels == class_names.index(class_name)).astype(int)
            class_probs = probs[:,class_names.index(class_name)]
            metrics[class_name + '_auc'] = roc_auc_score(classwise_labels, class_probs)
    return metrics


def compute_metrics_by_clip(cfg, frames_table_path, clips_table_path):
    '''
    For a particular dataset, use predictions for each filename to create predictions for whole clips and save the
    resulting metrics.
    :param cfg: project config
    :param frames_table_path: Path to CSV of Dataframe linking filenames to labels
    :param clips_table_path: Path to CSV of Dataframe linking clips to labels
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    preprocessing_fn = get_preprocessing_function(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = frames_table_path.split('/')[-1].split('.')[0] + '_clips'

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_labels = clips_df['class']
    clip_names = clips_df['filename']
    clip_pred_classes = []
    avg_pred_probs = np.zeros((clips_df.shape[0], len(cfg['DATA']['CLASSES'])))
    for i in range(len(clip_names)):

        # Get records from all files from this clip
        clip_name = clip_names[i].split('/')[-1].split(' ')[0]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        print("Making predictions for clip " + clip_name)

        # Make predictions for each image
        pred_classes, pred_probs = predict_set(model, preprocessing_fn, clip_files_df)

        # Compute average prediction probabilities for entire clip
        avg_pred_prob = np.mean(pred_probs, axis=0)
        avg_pred_probs[i] = avg_pred_prob

        # Record predicted class
        clip_pred = np.argmax(avg_pred_prob)
        clip_pred_classes.append(clip_pred)

    metrics = compute_metrics(cfg, np.array(clip_labels), np.array(clip_pred_classes), avg_pred_probs)
    print(metrics)
    doc = json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'clips_' + set_name + '.json', 'w'))

    # Save predictions
    avg_pred_probs_df = pd.DataFrame(avg_pred_probs, columns=cfg['DATA']['CLASSES'])
    avg_pred_probs_df.insert(0, 'filename', clips_df['filename'])
    avg_pred_probs_df.insert(1, 'class', clips_df['class'])
    avg_pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return


def compute_metrics_by_frame(cfg, dataset_files_path):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    preprocessing_fn = get_preprocessing_function(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'

    files_df = pd.read_csv(dataset_files_path)
    frame_labels = files_df['Class']    # Get ground truth

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(model, preprocessing_fn, files_df)

    # Compute and save metrics
    metrics = compute_metrics(cfg, np.array(frame_labels), np.array(pred_classes), pred_probs)
    doc = json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'frames_' + set_name + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'Frame Path', files_df['Frame Path'])
    pred_probs_df.insert(1, 'Class', files_df['Class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return


def b_line_threshold_curve(frame_preds_path, min_b_lines, max_b_lines, document=False):
    '''
    Varies the levels of thresholds for number of predicted frames with B-lines needed to classify a clip as
    pathological. Computes metrics for each threshold value. Save a table and visualization of the results.
    :param frame_preds_path: Path to CSV file containing frame-wise predictions
    :min_b_lines: minimum clip-wise B-line classification threshold
    :max_b_lines: maximum clip-wise B-line classification threshold
    :document: if set to True, generates a visualization and saves it as an image, along with a CSV
    '''

    N_A_LINES = '# A-line'
    N_B_LINES = '# B-line'
    ACCURACY = 'Accuracy'
    PRECISION_A_LINES = 'Precision (A-lines)'
    PRECISION_B_LINES = 'Precision (B-lines)'
    RECALL_A_LINES = 'Recall (A-lines)'
    RECALL_B_LINES = 'Recall (B-lines)'
    B_LINE_THRESHOLD = 'B-line Threshold'
    metrics_columns = [B_LINE_THRESHOLD, ACCURACY, PRECISION_A_LINES, PRECISION_B_LINES, RECALL_A_LINES, RECALL_B_LINES]

    preds_df = pd.read_csv(frame_preds_path)
    preds_df['Clip'] = preds_df["Frame Path"].str.rpartition("_")[0]
    preds_df['Pred Class'] = preds_df['b_lines'].ge(preds_df['a_lines']).astype(int)

    clips_df = preds_df.groupby('Clip').agg({'Class': 'first', 'Pred Class': 'sum'}).rename(columns={'Pred Class': N_B_LINES})
    metrics_df = pd.DataFrame()

    for threshold in range(min_b_lines, max_b_lines + 1):
        clips_df['Pred Class'] = clips_df[N_B_LINES].ge(threshold).astype(int)
        metrics = compute_metrics(cfg, np.array(clips_df['Class']), np.array(clips_df['Pred Class']))
        metrics_flattened = pd.json_normalize(metrics, sep='_')
        metrics_df = pd.concat([metrics_df, metrics_flattened], axis=0)

    metrics_df[B_LINE_THRESHOLD] = np.arange(min_b_lines, max_b_lines + 1)
    metrics_df.set_index(B_LINE_THRESHOLD, inplace=True)

    if document:
        # TODO: generate and save plot
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'b-line_thresholds_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return metrics_df


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    # dataset_path = cfg['PATHS']['EXT_VAL_FRAME_TABLE']
    # clips_path = cfg['PATHS']['EXT_VAL_CLIPS_TABLE']
    # compute_metrics_by_clip(cfg, dataset_path, clips_path)
    # compute_metrics_by_frame(cfg, dataset_path)
    b_line_threshold_curve('results/experiments/ext_frames_predictions20210617-164406_cropped_VGG16.csv', 1, 40)