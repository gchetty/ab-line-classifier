import dill
import yaml
import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.visualization.visualization import *
from src.models.models import get_model

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

# Repeated column names
B_LINE_THRESHOLD = 'B-line Threshold'
PRED_CLASS = 'Pred Class'
CLASS_NUM = 'Class'
CLIP = 'Clip'

# Class map
CLASS_IDX_MAP = dill.load(open(cfg['PATHS']['CLASS_NAME_MAP'], 'rb'))
IDX_CLASS_MAP = {v: k for k, v in CLASS_IDX_MAP.items()}  # Reverse the map


def predict_set(model, preprocessing_func, predict_df, threshold=0.5):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_func: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :param threshold: Classification threshold
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

    # Obtain prediction probabilities
    p = model.predict_generator(generator)
    test_predictions = (p[:, CLASS_IDX_MAP['b_lines']] >= threshold).astype(int)

    # Get prediction classes in original labelling system
    pred_classes = [IDX_CLASS_MAP[v] for v in list(test_predictions)]
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

    precision = precision_score(labels, preds, average='binary')
    recalls = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average='binary')

    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['precision'] = precision
    metrics['recall'] = recalls[CLASS_IDX_MAP['b_lines']]          # Recall of the positive class (i.e. sensitivity)
    metrics['specificity'] = recalls[CLASS_IDX_MAP['a_lines']]     # Specificity is recall of the negative class
    metrics['f1'] = f1
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


def compute_metrics_by_clip(cfg, frames_table_path, clips_table_path, class_thresh=0.5, cont_thresh=None):
    '''
    For a particular dataset, use predictions for each filename to create predictions for whole clips and save the
    resulting metrics.
    :param cfg: project config
    :param frames_table_path: Path to CSV of Dataframe linking filenames to labels
    :param clips_table_path: Path to CSV of Dataframe linking clips to labels
    :param class_thresh: Classification threshold for frame prediction
    :param cont_thresh: Contiguity threshold
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
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
        clip_name = clip_names[i]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        print("Making predictions for clip " + clip_name)

        # Make predictions for each image
        pred_classes, pred_probs = predict_set(model, preprocessing_fn, clip_files_df, threshold=class_thresh)

        # Compute average prediction probabilities for entire clip
        if cont_thresh:
            avg_pred_prob = highest_contiguous_pred_prob(pred_probs, ct=cont_thresh)
        else:
            avg_pred_prob = np.mean(pred_probs, axis=0)
        avg_pred_probs[i] = avg_pred_prob

        # Record predicted class
        clip_pred = (avg_pred_prob[CLASS_IDX_MAP['b_lines']] >= class_thresh).astype(int)
        clip_pred_classes.append(clip_pred)

    metrics = compute_metrics(cfg, np.array(clip_labels), np.array(clip_pred_classes), avg_pred_probs)
    print(metrics)
    json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'clips_' + set_name +
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    avg_pred_probs_df = pd.DataFrame(avg_pred_probs, columns=cfg['DATA']['CLASSES'])
    avg_pred_probs_df.insert(0, 'filename', clips_df['filename'])
    avg_pred_probs_df.insert(1, 'class', clips_df['class'])
    avg_pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return


def compute_metrics_by_frame(cfg, dataset_files_path, class_thresh=0.5):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param class_thresh: Classification threshold for frame prediction
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'

    files_df = pd.read_csv(dataset_files_path)
    frame_labels = files_df['Class']    # Get ground truth

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(model, preprocessing_fn, files_df, threshold=class_thresh)

    # Compute and save metrics
    metrics = compute_metrics(cfg, np.array(frame_labels), np.array(pred_classes), pred_probs)
    json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'frames_' + set_name +
                            datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'Frame Path', files_df['Frame Path'])
    pred_probs_df.insert(1, 'Class', files_df['Class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return


def b_line_threshold_experiment(frame_preds_path, min_b_lines, max_b_lines, class_thresh=0.5, contiguous=True, document=False):
    '''
    Varies the levels of thresholds for number of predicted frames with B-lines needed to classify a clip as
    pathological. Computes metrics for each threshold value. Saves a table and visualization of the results.
    :param frame_preds_path: Path to CSV file containing frame-wise predictions.
    :min_b_lines: minimum clip-wise B-line classification threshold
    :max_b_lines: maximum clip-wise B-line classification threshold
    :param class_thresh: Classification threshold for frame prediction
    :contiguous: if set to True, uses the maximum contiguous B-line predictions as the clip prediction threshold;
                 if set to False, uses the total B-line predictions as the clip prediction threshold.
    :document: if set to True, generates a visualization and saves it as an image, along with a CSV
    '''

    preds_df = pd.read_csv(frame_preds_path)
    preds_df[CLIP] = preds_df["Frame Path"].str.rpartition("_")[0]
    preds_df[PRED_CLASS] = preds_df['b_lines'].ge(class_thresh).astype(int)
    preds_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'preds.csv', index=False)

    if contiguous:
        n_b_lines_col = 'Contiguous Predicted B-lines'
        clips_df = preds_df.groupby(CLIP).agg({CLASS_NUM: 'max', PRED_CLASS: max_contiguous_b_line_preds})
    else:
        n_b_lines_col = 'Total Predicted B-lines'
        clips_df = preds_df.groupby(CLIP).agg({CLASS_NUM: 'max', PRED_CLASS: 'sum'})
    clips_df.rename(columns={PRED_CLASS: n_b_lines_col}, inplace=True)

    metrics_df = pd.DataFrame()
    tprs = []   # True positive rates at different CTs
    fprs = []   # False positive rates at different CTs

    for threshold in range(min_b_lines, max_b_lines + 1):
        clips_df[PRED_CLASS] = clips_df[n_b_lines_col].ge(threshold).astype(int)
        metrics = compute_metrics(cfg, np.array(clips_df[CLASS_NUM]), np.array(clips_df[PRED_CLASS]))
        metrics_flattened = pd.json_normalize(metrics, sep='_')
        metrics_df = pd.concat([metrics_df, metrics_flattened], axis=0)
        tprs.append(metrics['recall'])
        fprs.append(1. - metrics['specificity'])
    metrics_df.insert(0, B_LINE_THRESHOLD, np.arange(min_b_lines, max_b_lines + 1))

    if document:
        plot_b_line_threshold_experiment(metrics_df, min_b_lines, max_b_lines)
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'b-line_thresholds_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=False)
        clips_df.drop(PRED_CLASS, axis=1, inplace=True)
        clips_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'clip_contiguous_preds_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=True)
        plot_b_line_threshold_roc_curve(tprs, fprs)
    return metrics_df


def max_contiguous_b_line_preds(pred_series):
    '''
    Given a series of class predictions, determine the maximum number of contiguous B-line predictions
    :param pred_series: Pandas series of frame-wise integer predictions
    :return: Maximum contiguous B-line predictions in the series
    '''
    pred_arr = np.asarray(pred_series)
    max_contiguous = cur_contiguous = 0
    for i in range(pred_arr.shape[0]):
        if pred_arr[i] == 1:
            cur_contiguous += 1
        else:
            cur_contiguous = 0
        if cur_contiguous > max_contiguous:
            max_contiguous = cur_contiguous
    return max_contiguous


def highest_contiguous_pred_prob(pred_probs, ct):
    '''
    Determines the highest average frame prediction over ct contiguous clips
    :param pred_probs [N, C]: framewise predictions for each class
    :param ct: contiguity threshold
    '''
    max_b_pred = 0.0
    for i in range(0, pred_probs.shape[0] - ct + 1):
        avg_b_pred = np.mean(pred_probs[i:i + ct, 1])
        if avg_b_pred > max_b_pred:
            max_b_pred = avg_b_pred
    return np.array([1. - max_b_pred, max_b_pred])


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    frames_path = 'data/frames_external_a_b.csv' #'data/partitions/test_set_final.csv'
    clips_path = 'data/clips_by_patient_external_a_b.csv' #cfg['PATHS']['EXT_VAL_CLIPS_TABLE']
    #compute_metrics_by_clip(cfg, frames_path, clips_path, class_thresh=0.9, cont_thresh=None)
    #compute_metrics_by_frame(cfg, frames_path, class_thresh=0.9)
    b_line_threshold_experiment('results/experiments/exp5/external_frames_preds_all.csv', 0, 40, class_thresh=1.0, contiguous=False, document=True)