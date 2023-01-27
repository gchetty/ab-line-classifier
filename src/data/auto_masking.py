"""
This software is used to scrub raw clips of all on-screen information
(e.g. vendor logos, battery indicators, index mark, depth markers)
extraneous to the ultrasound beam itself. This is dedicated deep learning
masking software for ultrasound (AutoMask, WaveBase Inc., Waterloo, Canada).
"""

import cv2
import numpy as np
import os
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import img_to_array
import glob
import argparse
import matplotlib.pyplot as plt

class UnetSegmentation:

    def __init__(self):
        pass
    
    def get_bounding_box(self, binary_mask):
        i, j = np.where(binary_mask)
        return [min(i), max(i) , min(j), max(j)]

    def predict(self, input_paths, output_path, model_path, output_format='jpg', edge_preserve=0.95, save_cropped_roi=False):
        model = load_model(model_path)
        video_files = glob.glob(input_paths + "/**/*.mp4", recursive=True)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(os.path.join(output_path, 'bad_clips')):
            os.mkdir(os.path.join(output_path, 'bad_clips'))
        num_clips = len(video_files)
        for clip_index, file in enumerate(video_files):
            head, tail = os.path.split(file)
            # Creates new folder to store data to output_path
            try:
                output_folder_new = os.path.join(output_path, tail[:-4])
                if not os.path.exists(os.path.join(output_folder_new)):
                    os.mkdir(output_folder_new)
            except OSError:
                print('failed to create output folders')
                continue
            cap = cv2.VideoCapture(file)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
            print('FRS', frame_rate)

            video = cv2.VideoWriter(os.path.join(output_folder_new, tail[:-4] + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                    frame_rate, (frame_width, frame_height), True)
            step = max(int(num_frames * 0.1), 1)
            average_mask = np.zeros((frame_height, frame_width))
            num_iter = 0
            kernel_size_s = max(int(frame_height * (1 - edge_preserve)), 3)
            kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_s, kernel_size_s))
            kernel_size_b = max(int(frame_height * 0.05), 3)  # smoothing factor
            kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_b, kernel_size_b))

            for fn in range(0, num_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                res, frame = cap.read()
                if frame.shape[2] > 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = img_to_array(frame)
                img = resize(img, (128, 128, 1), mode='constant', preserve_range=True)

                img = img / 255.0
                img = np.expand_dims(img, 0)
                result = model.predict(img)
                result = np.squeeze(result, axis=0)
                result = np.squeeze(result, axis=2)
                mask = np.zeros((128, 128), dtype='uint8')
                mask[result > 0.4] = 255
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel_s, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_b, iterations=1)
                # Binarizes mask
                ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                mask[mask > 128] = 1
                average_mask = average_mask + mask
                num_iter += 1

            kernel = np.ones((5, 5), np.float32) / 25
            average_mask = cv2.filter2D(average_mask, -1, kernel)
            average_mask[average_mask < num_iter / 2] = 0
            average_mask[average_mask > num_iter / 2] = 1
            indices = self.get_bounding_box(average_mask)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_index in range(0, num_frames):
                print(f' clip {clip_index+1}/{num_clips} :::::::: frame{frame_index}/{num_frames}')
                ret, frame = cap.read()
                rgb = frame.copy()
                if output_format == 'jpg':
                    final_image =  np.array(cv2.bitwise_and(frame, frame, mask=average_mask.astype(np.uint8))).astype(np.uint8)
                    if save_cropped_roi:
                        final_image = final_image[indices[0]:indices[1], indices[2]: indices[3]]
                    cv2.imwrite(os.path.join(output_folder_new, str(frame_index) + '.jpg'), final_image)
                elif output_format == 'mp4':
                    res = np.array(cv2.bitwise_and(rgb, rgb, mask=average_mask.astype(np.uint8))).astype(np.uint8)
                    video.write(res)
            cv2.imwrite(os.path.join(output_folder_new, 'mask.jpg'), average_mask * 255)
            video.release()



if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('-i', '--input_path', required=True, help='path to folder containing ultra sound clips for masking')
    ap.add_argument('-o', '--output_path', required=True, help='path to folder where masking results will be stored ')
    ap.add_argument('-m', '--model_path', required=True, help='path to the ML model to be used for prediction')
    ap.add_argument('-f', '--output_format', required=True, help='format of output files; either jpg or mp4 ')
    ap.add_argument('-e', '--edge_preserve', required=False, help='a float [0 1] presenting edge preservation. '
                                                                  '1 keep edges unchanged. Values below 0.5 will eventually '
                                                                  'remove most of beam area. Good value is around 0.95')
    ap.add_argument('-c', '--save_cropped_roi', required=False, help='If True, only the cropped ROI is saved')


    args = vars(ap.parse_args())

    input_paths = args['input_path']
    output_path = args['output_path']
    model_path = args['model_path']
    output_format = args['output_format']
    edge_preserve = args['edge_preserve']
    save_cropped_roi = args['save_cropped_roi']
    if edge_preserve and (float(edge_preserve) < 0 or float(edge_preserve) > 1):
        raise ValueError('edge_preserve has to be in [0 1]')

    unet_seg = UnetSegmentation()
    unet_seg.predict(input_paths, output_path, model_path, output_format=output_format, edge_preserve=float(edge_preserve),
                     save_cropped_roi=save_cropped_roi)
