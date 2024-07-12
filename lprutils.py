from PIL import Image
import io
import base64
import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np
import cv2
from lprtools.vision import draw_bbox, draw_text

def display_plates(img, plates_info, resize=(1000, 600)):
    for plate in plates_info:
        coordinates = plate['coordinates']
        draw_bbox(img, np.array(coordinates))
        plate_number = plate["plate_number"]
        cv2.putText(img, plate_number, (int(coordinates[0]), int(coordinates[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        img = cv2.resize(img, resize)
    return img

def get_image_size_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read())
        return len(image_data)
    
class Logger:
    def __init__(self, name='lpr_test', log_file='./logs/lpr_test.log', level=logging.INFO):
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_logger(self):
        return self.logger
        
def load_image(img_content, mode):
    if mode == 'path':
        img = cv2.imread(img_content)
        if img is None:
            raise ValueError(f"Image at path {img_content} could not be loaded.")
        return img
    else:
        img_data = base64.b64decode(img_content)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    
    
if __name__ == "__main__":
    # img_path = 'tmp.jpg'
    # base64_size = get_image_size_base64(img_path)
    # print(f"The base64 encoded size is {base64_size} bytes.")
    logger = Logger().get_logger()
    logger.info("This is an info message")
    