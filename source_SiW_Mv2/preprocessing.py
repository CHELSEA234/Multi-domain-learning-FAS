# -*- coding: utf-8 -*-
# Copyright 2022
# 
# Multi-domain Learning for Updating Face Anti-spoofing Models (ECCV 2022)
# Xiao Guo, Yaojie Liu, Anil Jain, and Xiaoming Liu
# 
# All Rights Reserved.s
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
import face_alignment
import glob
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import io
import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def video_process(vlist, folder_dir='./test/'):
    for vd in tqdm(vlist):
        folder_name = vd.split('/')[-1].split('.')[0]
        folder = os.path.join(folder_dir, folder_name)

        if not os.path.exists(folder):
            os.makedirs(folder)    

        cap = cv2.VideoCapture(vd)
        fr = 1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = Image.fromarray(frame)
                width, height = frame.size
                scale = 1.0
                if max(width, height) > 800:
                    scale = 800.0 / max(width, height)
                    detect_frame = frame.resize((int(width*scale),int(height*scale)), Image.Resampling.BICUBIC)
                    detect_frame = np.array(detect_frame)
                else:
                    detect_frame = np.array(frame)

                scale = 1/scale
                preds = fa.get_landmarks(detect_frame)
                frame = np.array(frame)
                if frame is None:
                    # This shouldn't happen as we should only be here when ret == True
                    raise Exception("Failed to read source video")
                frame_shape = frame.shape

                if preds is None:
                    print(f'No Face found at frame #{fr} of {vd} (frame # won\'t be incremented)')
                    continue
                else:
                    pred = (preds[0] * [scale, scale]).astype(int)
                    if len(preds) > 1:
                        biggest_eye2eye_dis = -100
                        for test_pred in preds:
                            test_pred = (test_pred * [scale, scale]).astype(int)
                            eye2eye_dis = np.sqrt(np.sum(np.square(
                                np.abs(test_pred[36, :] - test_pred[45, :])
                            ))) / 2
                            if eye2eye_dis > biggest_eye2eye_dis:
                                pred = test_pred
                                biggest_eye2eye_dis = eye2eye_dis

                eye2eye_dis = np.sqrt(np.sum(np.square(
                    np.abs(pred[36, :] - pred[45, :])
                ))) / 2
                nose_len = np.sqrt(np.sum(np.square(
                    np.abs(pred[27, :] - pred[30, :])
                ))) / 2
                face_len = np.sqrt(np.sum(np.square(
                    np.abs(pred[27, :] - pred[8, :])
                ))) / 2
                if face_len == 0.0:
                    nose_face_ratio = 1.0 # Chin is on nose, ie. spoof
                else:
                    nose_face_ratio = nose_len / face_len

                eye_center = (pred[36, :] + pred[45, :]) / 2

                xl = int(eye_center[0] - eye2eye_dis * 2.3)
                xr = int(eye_center[0] + eye2eye_dis * 2.3)
                yt = int(eye_center[1] - eye2eye_dis * 1.6)
                yb = int(eye_center[1] + eye2eye_dis * 3.0)

                if xl < 0 or yt < 0 or xr >= frame.shape[1] or yb >= frame.shape[0]:
                    (xl_pad, xr_pad, yt_pad, yb_pad) = (0,0,0,0)
                    if xl < 0:
                      xl_pad = abs(xl)
                    if yt < 0:
                      yt_pad = abs(yt)
                    if xr > (frame.shape[1] - 1):
                      xr_pad = xr - frame.shape[1] + 1
                    if yb > (frame.shape[0] - 1):
                      yb_pad = yb - frame.shape[0] + 1

                    large_fr = np.zeros((yt_pad + yb_pad + frame.shape[0],
                                         xl_pad + xr_pad + frame.shape[1],
                                         3))
                    large_fr[yt_pad:yt_pad + frame.shape[0],
                             xl_pad:xl_pad + frame.shape[1],
                             :] = frame
                    xl += xl_pad
                    xr += xl_pad
                    yt += yt_pad
                    yb += yt_pad
                    face = large_fr[yt:yb, xl:xr, :]
                else:
                    face = frame[yt:yb, xl:xr, :]

                x_scale = float(256) / float(xr-xl)
                y_scale = float(256) / float(yb-yt)

                pred[:, 0] = pred[:, 0] - int(eye_center[0] - eye2eye_dis * 2.3)
                pred[:, 1] = pred[:, 1] - int(eye_center[1] - eye2eye_dis * 1.6)
                face = Image.fromarray(face.astype(np.uint8)).resize((256, 256), Image.Resampling.BICUBIC)
                pred = (pred * [x_scale,y_scale]).astype(int)

                face_numpy_array = np.array(face)
                img_rgb = face_numpy_array
                fname = folder + '/' + str(fr) + '.png'
                lmname = folder + '/' + str(fr) + '.npy'
                cv2.imwrite(fname, img_rgb)
                np.save(lmname, pred)
                fr += 1
            else:
                break

if __name__ == "__main__":
    file_path = './sample_video/live/*.mov'
    video_list = glob.glob(file_path)
    video_process(video_list, folder_dir='./demo/live/')
    file_path = './sample_video/spoof/*.mov'
    video_list = glob.glob(file_path)
    video_process(video_list, folder_dir='./demo/spoof/')