import dill
import time
import yaml
import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.models import load_model
import onnx
import cv2
from onnx_tf.backend import prepare

from src.visualization.visualization import *
from src.models.models import get_model
from src.deploy import AB_classifier_preprocess
from src.data.preprocessor import Preprocessor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

# Repeated column names
B_LINE_THRESHOLD = 'B-line Threshold'
PRED_CLASS = 'Pred Class'
CLASS_NUM = 'Class'
CLIP = 'Clip'
B_PROB = 'b_lines'
A_PROB = 'a_lines'
SLIDING_WINDOW = 'Sliding Window Length'

# Class map
CLASS_IDX_MAP = dill.load(open(cfg['PATHS']['CLASS_NAME_MAP'], 'rb'))
IDX_CLASS_MAP = {v: k for k, v in CLASS_IDX_MAP.items()}  # Reverse the map

# MODEL FORMAT
model_ext = os.path.splitext(cfg['PATHS']['MODEL_TO_LOAD'])[1]
ONNX = True if model_ext == '.onnx' else False


def restore_model(model_path):
    '''
    Restores the model from serialized weights
    param model_path: Path at which weights are stored
    return: keras Model object
    '''
    # Restore the model from serialized weights
    model_ext = os.path.splitext(model_path)[1]
    if model_ext == '.onnx':
        model = prepare(onnx.load(model_path))
    else:
        model = load_model(model_path, compile=False)
    return model


def predict_set(model, preprocessing_func, predict_df, onnx=False, threshold=0.5):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_func: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :param onnx: True if model was restored from a .onnx file
    :param threshold: Classification threshold
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    if onnx:
        p = np.zeros((predict_df.shape[0], 2))
        for i in range(predict_df.shape[0]):
            frame_path = os.path.join(cfg['PATHS']['FRAMES'], predict_df.loc[i, 'Frame Path'])
            frame = cv2.imread(frame_path)
            frame = np.expand_dims(frame, axis=0)
            preprocessed_frame = AB_classifier_preprocess(frame, preprocessing_func)
            p[i] = model.run(preprocessed_frame).output
    else:
        frames_dir = cfg['PATHS']['FRAMES_DIR']
        dataset = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in predict_df['Frame Path'].tolist()], predict_df['Class']))
        preprocessor = Preprocessor(preprocessing_func)
        preprocessed_set = preprocessor.prepare(dataset, shuffle=False, augment=False)

        # Obtain prediction probabilities
        p = model.predict(preprocessed_set)

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


def compute_clip_predictions(cfg, frames_table_path, clips_table_path, class_thresh=0.5, clip_algorithm='contiguous', calculate_metrics=True):
    '''
    For a particular dataset, use predictions for each filename to create predictions for whole clips and save the
    resulting metrics.
    :param cfg: project config
    :param frames_table_path: Path to CSV of Dataframe linking filenames to labels
    :param clips_table_path: Path to CSV of Dataframe linking clips to labels
    :param class_thresh: Classification threshold for frame prediction
    :param clip_algorithm: Choice of clip prediction algorithm (one of 'contiguous', 'average', 'sliding_window')
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    model = restore_model(cfg['PATHS']['MODEL_TO_LOAD'])
    set_name = frames_table_path.split('/')[-1].split('.')[0] + '_clips'

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_pred_classes = []
    all_pred_probs = np.zeros((clips_df.shape[0], len(cfg['DATA']['CLASSES'])))
    print("Found {} clips. Determining clip predictions with {} algorithm.".format(len(clip_names), clip_algorithm))
    for i in range(len(clip_names)):

        # Get records from all files from this clip
        clip_name = clip_names[i]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        print("Making predictions for clip " + clip_name)

        # Make predictions for each image
        pred_classes, pred_probs = predict_set(model, preprocessing_fn, clip_files_df, onnx=ONNX, threshold=class_thresh)

        # Compute average prediction for entire clip
        if clip_algorithm == 'contiguous':
            clip_pred_prob = predict_with_contiguity_threshold(pred_probs, cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD'], class_thresh)
        elif clip_algorithm == 'sliding_window':
            clip_pred_prob = highest_avg_contiguous_pred_prob(pred_probs, cfg['CLIP_PREDICTION']['SLIDING_WINDOW'])
        elif clip_algorithm =='average':
            clip_pred_prob = np.mean(pred_probs, axis=0)
        else:
            raise Exception('Unknown value for "clip_algorithm" argument.')
        all_pred_probs[i] = clip_pred_prob

        # Record predicted class
        pred_class = (clip_pred_prob[CLASS_IDX_MAP['b_lines']] >= class_thresh).astype(int)
        clip_pred_classes.append(pred_class)

    if calculate_metrics:
        clip_labels = clips_df['class']
        if clip_algorithm != 'contiguous':
            metrics = compute_metrics(cfg, np.array(clip_labels), np.array(clip_pred_classes), all_pred_probs)
        else:
            metrics = compute_metrics(cfg, np.array(clip_labels), np.array(clip_pred_classes))
        json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'clips_' + set_name +
                                 datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(all_pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'filename', clips_df['filename'])
    pred_probs_df.insert(1, 'class', clips_df['class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return pred_probs_df


def compute_frame_predictions(cfg, dataset_files_path, class_thresh=0.5, calculate_metrics=True):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param class_thresh: Classification threshold for frame prediction
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    model = restore_model(cfg['PATHS']['MODEL_TO_LOAD'])
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'

    files_df = pd.read_csv(dataset_files_path)

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(model, preprocessing_fn, files_df, onnx=ONNX, threshold=class_thresh)

    # Compute and save metrics
    if calculate_metrics:
        frame_labels = files_df['Class']  # Get ground truth
        metrics = compute_metrics(cfg, np.array(frame_labels), np.array(pred_classes), pred_probs)
        json.dump(metrics, open(cfg['PATHS']['METRICS'] + 'frames_' +
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'Frame Path', files_df['Frame Path'])
    pred_probs_df.insert(1, 'Class', files_df['Class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + '_predictions' +
                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return pred_probs_df


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
    preds_df[PRED_CLASS] = preds_df[B_PROB].ge(class_thresh).astype(int)
    preds_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'preds.csv', index=False)

    if contiguous:
        n_b_lines_col = 'Contiguous Predicted B-lines'
        clips_df = preds_df.groupby(CLIP).agg({CLASS_NUM: 'max', PRED_CLASS: max_contiguous_b_line_preds_from_series})
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
        plot_b_line_threshold_experiment(metrics_df, min_b_lines, max_b_lines, B_LINE_THRESHOLD, class_thresh)
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'b-line_thresholds_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=False)
        clips_df.drop(PRED_CLASS, axis=1, inplace=True)
        clips_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'clip_contiguous_preds_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=True)
        plot_b_line_threshold_roc_curve(tprs, fprs)
    return metrics_df

def max_contiguous_b_line_preds_from_series(pred_series):
    '''
    Given a Pandas series of class predictions, determine the maximum number of contiguous B-line predictions
    :param preds: Numpy array of frame-wise integer predictions
    :return: Maximum contiguous B-line predictions in the series
    '''
    pred_arr = np.asarray(pred_series)
    return max_contiguous_b_line_preds(pred_arr)

def max_contiguous_b_line_preds(preds):
    '''
    Given a numpy array of class predictions, determine the maximum number of contiguous B-line predictions
    :param preds: Numpy array of frame-wise integer predictions
    :return: Maximum contiguous B-line predictions in the series
    '''
    max_contiguous = cur_contiguous = 0
    for i in range(preds.shape[0]):
        if preds[i] == 1:
            cur_contiguous += 1
        else:
            cur_contiguous = 0
        if cur_contiguous > max_contiguous:
            max_contiguous = cur_contiguous
    return max_contiguous

def predict_with_contiguity_threshold(pred_probs, contiguity_threshold, classification_threshold):
    '''
    Determine prediction probabilities using the contiguous frames method
    :param pred_probs: Numpy array of prediction probabilities
    :param contiguity_threshold: The contiguity threshold
    :param classification_threshold: The classification threshold
    '''
    b_preds = (pred_probs[:,1] > classification_threshold).astype(int)
    clip_pred = int(max_contiguous_b_line_preds(b_preds) >= contiguity_threshold)
    return np.array([1 - clip_pred, clip_pred])

def predict_clipwise_with_contiguity_threshold_wb(preds, target_class, contiguity_threshold, classification_threshold):
    '''
    Determine prediction probabilities using the contiguous frames method (from WaveBase output)
    :param preds: Pandas array of prediction probabilities
    :param target_class: Target class for prediction
    :param contiguity_threshold: The contiguity threshold
    :param classification_threshold: The classification threshold
    '''
    cur_contiguous = 0
    for i in range(preds.shape[0]):
        if preds.loc[i, 0] == target_class and float(preds.loc[i, 1]) > classification_threshold:
            cur_contiguous += 1
        else:
            cur_contiguous = 0
        if cur_contiguous >= contiguity_threshold:
            return True
    return False

def compute_clip_predictions_wb(cfg):
    '''
    Loads the frame-wise prediction csvs exported from the WaveBase device and creates clip-wise predictions using the contiguous frames method
    :param cfg: project config
    '''
    # Point to acquired data
    rootdir = cfg['PATHS']['RT_ROOT_DIR']
    dated_dirs = next(os.walk(rootdir))[1]
    recording_dir = 'recordings'

    res = []
    # Loop over all directories containing csvs
    for dated_dir in dated_dirs:
        for root, dir, files in os.walk(os.path.join(rootdir, dated_dir, recording_dir)):
            # Keep only predicted probs csvs
            csvs = [file for file in files if ".csv" in file]
            for csv in csvs:
                clip_name = str.replace(csv, "_probs.csv", ".mkv")
                fname = os.path.join(root, csv)
                data = pd.read_csv(fname, delimiter=',', header=None, dtype=str)
                res.append([clip_name,
                            'B-Line'
                            if predict_clipwise_with_contiguity_threshold_wb(data,
                                                                    'B-Lines',
                                                                    cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD'],
                                                                    cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'])
                            else 'A-Line'])

    res_df = pd.DataFrame(res, columns=['filename', 'prediction'])
    res_file_name = 'results/predictions/'+ rootdir.split('/')[1] + '_clip_predictions_T' \
                    + str(cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD']) + '_t0' \
                    + str(cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'])[2] \
                    + '_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'
    res_df.to_csv(res_file_name, index=False)
    return res_df


def highest_avg_contiguous_pred_prob(pred_probs, window_length):
    '''
    Determines the highest average frame prediction over ct contiguous clips
    :param pred_probs [N, C]: framewise predictions for each class
    :param window_length: Length of sliding window for average contiguous predictions
    '''
    max_b_pred = 0.0
    for i in range(0, pred_probs.shape[0] - window_length + 1):
        avg_b_pred = np.mean(pred_probs[i:i + window_length, 1])
        if avg_b_pred > max_b_pred:
            max_b_pred = avg_b_pred
    return np.array([1. - max_b_pred, max_b_pred])


def sliding_window_variation_experiment(frame_preds_path, min_window_length, max_window_length, class_thresh=0.5, document=False):
    '''
    Varies the sliding window length over which predictions are averaged when classifying clips using the highest
    average contiguous probability algorithm. Computes metrics for each threshold value.
    Saves a table and visualization of the results.
    :param frame_preds_path: Path to CSV file containing frame-wise predictions.
    :min_window_length: minimum sliding window length
    :max_window_length maximum sliding window length
    :param class_thresh: Classification threshold for frame prediction
    :document: if set to True, generates a visualization and saves it as an image, along with a CSV
    '''

    preds_df = pd.read_csv(frame_preds_path)
    preds_df[CLIP] = preds_df["Frame Path"].str.rpartition("_")[0]

    def highest_avg_b_line_prob(b_probs_series, window_length):
        '''
        Helper function that computes highest B-line probability averaged over window_length contiguous frames
        '''
        max_b_pred = 0.0
        for i in range(0, b_probs_series.shape[0] - window_length + 1):
            avg_b_pred = np.mean(b_probs_series[i:i + window_length])
            if avg_b_pred > max_b_pred:
                max_b_pred = avg_b_pred
        return max_b_pred

    metrics_df = pd.DataFrame()
    for window_length in range(min_window_length, max_window_length + 1):
        clips_df = preds_df.groupby(CLIP).agg({CLASS_NUM: 'max', B_PROB: lambda x: highest_avg_b_line_prob(x, window_length)})
        clips_df[A_PROB] = 1. - clips_df[B_PROB]
        clips_df[PRED_CLASS] = clips_df[B_PROB].ge(class_thresh).astype(int)

        pred_probs = np.array(clips_df[[A_PROB, B_PROB]])
        metrics = compute_metrics(cfg, np.array(clips_df[CLASS_NUM]), np.array(clips_df[PRED_CLASS]), probs=pred_probs)
        metrics_flattened = pd.json_normalize(metrics, sep='_')
        metrics_df = pd.concat([metrics_df, metrics_flattened], axis=0)
    metrics_df.insert(0, SLIDING_WINDOW, np.arange(min_window_length, max_window_length + 1))

    if document:
        plot_b_line_threshold_experiment(metrics_df, min_window_length, max_window_length, SLIDING_WINDOW, class_thresh)
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'sliding_window_exp_c' + str(class_thresh) + '_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=False)
        clips_df.drop(PRED_CLASS, axis=1, inplace=True)
        clips_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'clip_sliding_window_preds_c' + str(class_thresh) + '_' +
                              datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv', index=True)

def clock_avg_runtime(n_gpu_warmup_runs, n_experiment_runs):
    '''
    Measures the average inference time of a trained model. Executes a few warm-up runs, then measures the inference
    time of the model over a series of trials.
    :param n_gpu_warmup_runs: The number of inference runs to warm up the GPU
    :param n_experiment_runs: The number of inference runs to record
    :return: Average and standard deviation of the times of the recorded inference runs
    '''
    times = np.zeros((n_experiment_runs))
    img_dim = cfg['DATA']['IMG_DIM']

    model = restore_model(cfg['PATHS']['MODEL_TO_LOAD'])

    for i in range(n_gpu_warmup_runs):
        x = tf.random.normal((1, img_dim[0], img_dim[1], 3))
        y = model(x)
    for i in range(n_experiment_runs):
        x = tf.random.normal((1, img_dim[0], img_dim[1], 3))
        t_start = time.time()
        y = model(x)
        times[i] = time.time() - t_start
    t_avg_ms = np.mean(times) * 1000
    t_std_ms = np.std(times) * 1000
    print("Average runtime = {:.3f} ms, standard deviation = {:.3f} ms".format(t_avg_ms, t_std_ms))

if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    frames_path = cfg['PATHS']['FRAME_TABLE']
    clips_path = cfg['PATHS']['CLIPS_TABLE']
    compute_clip_predictions(cfg, frames_path, clips_path, class_thresh=cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'],
                             clip_algorithm=cfg['CLIP_PREDICTION']['ALGORITHM'], calculate_metrics=True)
    compute_frame_predictions(cfg, frames_path, class_thresh=0.5, calculate_metrics=True)
    #b_line_threshold_experiment('results/predictions/frame_preds_0.5c.csv', 0, 40, class_thresh=0.95, contiguous=False, document=True)
    #sliding_window_variation_experiment('results/predictions/frame_preds_0.5c.csv', 1, 40, class_thresh=0.95, document=True)
    #compute_clip_predictions_wb(cfg, target_class='B-Lines')