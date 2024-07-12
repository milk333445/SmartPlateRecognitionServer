import os
import cv2
import torch
import numpy as np
import argparse
import time
import copy
from lprtools.detect import DetectorBuilder
from lprtools.vision import draw_bbox, draw_text

        
class LPRDetector:
    def __init__(self, config_path, perspect_transform=False):
        self.detector = DetectorBuilder(config_path)
        self.keys = self.detector.model_names
        self.perspect_transform = perspect_transform
        self.car_char_cls = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 0, 27: 1, 28: 2, 29: 3, 30: 4, 31: 5, 32: 6, 33: 7, 34: 8, 35: 9}

    
    def process_images(self, images, mode='pr'):
        detection_results = self.detector.car_plate.run(images)
        print(detection_results)
        if len(detection_results) == 0:
            return []
        plates_info = {}
        all_plate_rois = []
        plate_coords = []

        for i, img in enumerate(images):
            plates_info[i] = []
            for result in detection_results[i]:
                x1, y1, x2, y2, conf, class_idx = int(result[0]), int(result[1]), int(result[2]), int(result[3]), float(result[4]), int(result[5])
                if self.perspect_transform:
                    plate_roi = self.apply_perspective_transform(img, x1, y1, x2, y2)
                else:
                    plate_roi = img[y1:y2, x1:x2]
                all_plate_rois.append(plate_roi)
                plate_coords.append((i, x1, y1, x2, y2, conf, class_idx))
        
        char_start_time = time.time()
        if len(all_plate_rois) == 0:
            return plates_info
        
        if mode == 'pr' or mode == 'pgs':
            char_results = self.detector.car_char.run(all_plate_rois)

            for idx, char_result in enumerate(char_results):
                if len(char_result) == 0:
                    continue
                image_index, x1, y1, x2, y2, conf, class_idx = plate_coords[idx]
                plate_number, carplate_confs, carplate_coords = self.extract_plate_number(char_result)
                plates_info[image_index].append({
                    'coordinates': [x1, y1, x2, y2, round(conf, 4), class_idx],
                    'plate_number': plate_number,
                    'carplate_confs': carplate_confs,
                    'carplate_coords': carplate_coords
                })
        elif mode == 'ppd':
            for idx, plate_roi in enumerate(all_plate_rois):
                image_index, x1, y1, x2, y2, conf, class_idx = plate_coords[idx]
                plates_info[image_index].append({
                    'coordinates': [x1, y1, x2, y2, round(conf, 4), class_idx],
                    'plate_number': '',
                    'carplate_confs': [],
                    'carplate_coords': []
                })
        
        return plates_info

                
    def apply_perspective_transform(self, img, x1, y1, x2, y2):
        src_corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
        dst_corners = np.array([[0, 0], [300, 0], [0, 100], [300, 100]])
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        return cv2.warpPerspective(img, M, (300, 100))

    def extract_plate_number(self, char_result):
        sorted_results = sorted(char_result, key=lambda x: x[0])
        carplate_coords = [[int(result[0]), int(result[1]), int(result[2]), int(result[3])] for result in sorted_results]
        carplate_confs = [round(float(result[4]), 4) for result in sorted_results]
        try:
            car_numbers = ''.join([str(self.detector.car_char_cls[int(sorted_result[5])]) for sorted_result in sorted_results])
        except Exception as e:
            car_numbers = ''.join([str(self.car_char_cls[int(sorted_result[5])]) for sorted_result in sorted_results])
        return car_numbers, carplate_confs, carplate_coords
    

    
if __name__ == "__main__":
    detectpr = LPRDetector('./configs/detection.yaml')
    img = cv2.imread('tmp.jpg')
    images = [img]
    plates_info = detectpr.process_images(images)
    print(plates_info)