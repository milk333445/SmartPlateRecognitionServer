import cv2
import time
import logging
import os


class LPRDataHandler:
    def __init__(self, save_path, conf_threshold, logger=None):
        self.save_path = save_path
        self.conf_threshold = conf_threshold
        self.logger = logger if logger else logging.getLogger(__name__)
        # self.initialize_directories()
        self.car_char_cls = {chr(65 + i) if i < 26 else str(i - 26): i for i in range(36)}
        
    def initialize_directories(self):
        try:
            self.dataset_path = os.path.join(self.save_path, "dataset")
            self.review_data_char_path = os.path.join(self.dataset_path, "review_data_char")
            self.review_data_path = os.path.join(self.dataset_path, "review_data")
            self.train_data_path = os.path.join(self.dataset_path, "train_data")
            self.train_images_path = os.path.join(self.train_data_path, "images")
            self.train_labels_path = os.path.join(self.train_data_path, "labels")
            self.review_data_multi_results_path = os.path.join(self.dataset_path, "review_data_multi_results")
            self.visualize_path = os.path.join(self.save_path, "visualize")
            self.visualize_review_char_path = os.path.join(self.visualize_path, "review_data_char")
            self.visualize_train_data_path = os.path.join(self.visualize_path, "train_data")
            self.visualize_review_data_multi_results_path = os.path.join(self.visualize_path, "review_data_multi_results")
            os.makedirs(self.dataset_path, exist_ok=True)
            os.makedirs(self.review_data_char_path, exist_ok=True)
            os.makedirs(self.review_data_path, exist_ok=True)
            os.makedirs(self.train_data_path, exist_ok=True)
            os.makedirs(self.review_data_multi_results_path, exist_ok=True)
            os.makedirs(self.visualize_path, exist_ok=True)
            os.makedirs(self.visualize_review_char_path, exist_ok=True)
            os.makedirs(self.visualize_train_data_path, exist_ok=True)
            os.makedirs(self.visualize_review_data_multi_results_path, exist_ok=True)
            os.makedirs(self.train_images_path, exist_ok=True)
            os.makedirs(self.train_labels_path, exist_ok=True)
            self.logger.info("All directories initialized successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while initializing directories: {e}")
            
    def handle_plate_data(self, plates_info, images):
        """
        Handles the plate data by processing the results and saving images based on certain conditions.

        plate_info = {
            "0": [
                {"coordinates": [736, 104, 885, 197, 0.9297, 0], 
                "plate_number": "3H7797", 
                "carplate_confs": [0.95, 0.96, 0.97, 0.98, 0.99, 0.95], 
                "carplate_coords": [[9, 30, 29, 70], [29, 30, 52, 71], [57, 31, 78, 73], [79, 32, 100, 74], [100, 35, 123, 75], [122, 34, 144, 77]]}
            ],
            "1": [
                {"coordinates": [736, 104, 885, 197, 0.9297, 0], 
                "plate_number": "3H7797", 
                "carplate_confs": [0.85, 0.82, 0.81, 0.84, 0.80, 0.83], 
                "carplate_coords": [[9, 30, 29, 70], [29, 30, 52, 71], [57, 31, 78, 73], [79, 32, 100, 74], [100, 35, 123, 75], [122, 34, 144, 77]]}
            ]
        }
        
        Args:
            plates_info (dict): A dictionary containing information about the detected plates.
            images (list): A list of images corresponding to the plates.

        Returns:
            None
        """
        self.logger.info("Handling plate data...")
        current_time = time.time()
        results = []
        for idx, plate_info in plates_info.items():
            # 如果一張圖片有一個以上的車牌，就要存到review_data_multi_results
            if len(plate_info) > 1:
                self.logger.info(f"Multiple results detected in image {idx}.")
                image = images[int(idx)]
                self.save_image(image, os.path.join(self.review_data_multi_results_path, f"multi_results_{current_time}_{idx}.jpg"))
                continue
            if len(plate_info) == 0:
                self.logger.info(f"No results detected in image {idx}.")
                image = images[int(idx)]
                self.save_image(image, os.path.join(self.review_data_path, f"no_bbox_{current_time}_{idx}.jpg"))
                continue
            
            for data in plate_info:
                plate_number = data.get('plate_number', '')
                coordinates = data.get('coordinates', [])
                carplate_confs = data.get('carplate_confs', [])
                carplate_coords = data.get('carplate_coords', [])
                image = images[int(idx)]
                # 如果沒有框到車牌，就要存到review_data
                if len(coordinates) == 0:
                    self.save_image(image, os.path.join(self.review_data_path, f"no_bbox_{current_time}_{idx}.jpg"))
                    continue
                
                # 如果沒有檢測到車牌，但是有框到車牌，就要存到review_data_char
                if len(plate_number) == 0:
                    self.logger.info(f"No plate number detected in image {idx}.")
                    cropped_image = self.crop_plate(image, coordinates)
                    self.save_image(cropped_image, os.path.join(self.review_data_char_path, f"no_plate_{current_time}_{idx}.jpg"))
                    continue
                
                results.append(
                    {
                        'index': idx,
                        'plate_number': plate_number,
                        'coordinates': coordinates,
                        'carplate_confs': carplate_confs,
                        'carplate_coords': carplate_coords,
                        'image': image.copy()
                    }
                )
        # 一定要只有兩張圖片，這邊要確認一下
        if len(results) != 2:
            self.logger.info("Only two images are supported for continuous learning.")
            return
        else:
            self.compare_plate_results(results)
            
    def compare_plate_results(self, results):
        """
        Compare the results of two plate recognition processes.

        Args:
            results (list): A list containing the results of two plate recognition processes.

        Returns:
            None
        """
        result1, result2 = results[0], results[1]
        plate_number1, plate_number2 = result1['plate_number'], result2['plate_number']
        coordinates1, coordinates2 = result1['coordinates'], result2['coordinates']
        confs1, confs2 = result1['carplate_confs'], result2['carplate_confs']
        carplate_coords1, carplate_coords2 = result1['carplate_coords'], result2['carplate_coords']
        image1, image2 = result1['image'], result2['image']
        
        if len(plate_number1) != len(plate_number2):
            self.logger.info("Plate numbers have different lengths.")
            self.save_crop_and_visualize(image1, coordinates1, plate_number1)
            self.save_crop_and_visualize(image2, coordinates2, plate_number2)
        else:
            # 如果兩張圖片的車牌數量一樣，就要比對車牌號碼
            self.logger.info("Plate numbers have the same length.")
            if plate_number1 == plate_number2:
                self.logger.info("Plate numbers are the same.")
                self.handle_matching_plate_numbers(confs1, confs2, image1, image2, plate_number1, coordinates1, coordinates2, carplate_coords1, carplate_coords2)
            else:
                self.logger.info("Plate numbers are different.")
                self.handle_different_plate_numbers(image1, image2, plate_number1, plate_number2, coordinates1, coordinates2, confs1, confs2, carplate_coords1, carplate_coords2)
            
    def handle_different_plate_numbers(self, image1, image2, plate_number1, plate_number2, coordinates1, coordinates2, confs1, confs2, carplate_coords1, carplate_coords2):
        """
        Handles cases where there are different plate numbers detected for two images.

        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.
            plate_number1 (str): The plate number detected in the first image.
            plate_number2 (str): The plate number detected in the second image.
            coordinates1 (list): The coordinates of the detected plate in the first image.
            coordinates2 (list): The coordinates of the detected plate in the second image.
            confs1 (list): The confidence scores of the detected plate in the first image.
            confs2 (list): The confidence scores of the detected plate in the second image.
            carplate_coords1 (list): The coordinates of the car plate in the first image.
            carplate_coords2 (list): The coordinates of the car plate in the second image.

        Returns:
            None

        Raises:
            None
        """
        new_plate_number = ''.join([p1 if c1 > c2 else p2 for p1, c1, p2, c2 in zip(plate_number1, confs1, plate_number2, confs2)])
        # 跟原本比較結果，如果不一樣表示需要重新訓練
        if new_plate_number != plate_number1:
            self.save_data_for_training(image1, coordinates1, new_plate_number, carplate_coords1)
        if new_plate_number != plate_number2:
            self.save_data_for_training(image2, coordinates2, new_plate_number, carplate_coords2)
                 
    def handle_matching_plate_numbers(self, confs1, confs2, image1, image2, plate_number, coordinates1, coordinates2, carplate_coords1, carplate_coords2):
        """
        Handles the matching plate numbers by checking the confidence scores of the two images.
        If the confidence score of an image is below the threshold, it saves the data for training.

        Parameters:
        - confs1 (list): List of confidence scores for image1.
        - confs2 (list): List of confidence scores for image2.
        - image1 (str): Path to image1.
        - image2 (str): Path to image2.
        - plate_number (str): Plate number.
        - coordinates1 (tuple): Coordinates of the plate in image1.
        - coordinates2 (tuple): Coordinates of the plate in image2.
        - carplate_coords1 (tuple): Coordinates of the car plate in image1.
        - carplate_coords2 (tuple): Coordinates of the car plate in image2.
        """
        train_image1 = any(conf < self.conf_threshold for conf in confs1)
        train_image2 = any(conf < self.conf_threshold for conf in confs2)
        
        if train_image1:
            self.save_data_for_training(image1, coordinates1, plate_number, carplate_coords1)
        if train_image2:
            self.save_data_for_training(image2, coordinates2, plate_number, carplate_coords2)
    
    def save_data_for_training(self, image, coordinates, plate_number, carplate_coords):
        current_time = time.time()
        cropped_image = self.crop_plate(image, coordinates)
        # transfer x1, y1, x2, y2 to yolo format
        carplate_coords_yolo = self.adjust_carplate_coords(carplate_coords, coordinates)
        # save image and label(txt)
        self.save_image(cropped_image, os.path.join(self.train_images_path, f"{current_time}.jpg"))
        self.save_label(plate_number, carplate_coords_yolo, os.path.join(self.train_labels_path, f"{current_time}.txt"))
      
        # visualize
        image = self.visualize_plate(image, coordinates, plate_number)
        self.save_image(image, os.path.join(self.visualize_train_data_path, f"{current_time}.jpg"))
      
    def save_label(self, plate_number, carplate_coords, path):
        with open(path, 'w') as file:
            for i, coord in enumerate(carplate_coords):
                file.write(f"{self.car_char_cls[plate_number[i]]} {' '.join(map(str, coord))}\n")
        
    def adjust_carplate_coords(self, carplate_coords, coordinates):
        """
        Adjusts the carplate coordinates to YOLO format.

        Args:
            carplate_coords (list): List of carplate coordinates in the format [x1, y1, x2, y2].
            coordinates (list): List of image coordinates in the format [x1, y1, x2, y2].

        Returns:
            list: List of adjusted carplate coordinates in YOLO format [x, y, w, h].
        """
        x1, y1, x2, y2 = coordinates[:4]
        width, height = x2 - x1, y2 - y1
        result = []
        for coord in carplate_coords:
            x1, y1, x2, y2 = coord
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            result.append([x / width, y / height, w / width, h / height])
        return result
        
    def save_crop_and_visualize(self, image, coordinates, plate_number):
        current_time = time.time()
        cropped_image = self.crop_plate(image, coordinates)
        self.save_image(cropped_image, os.path.join(self.review_data_char_path, f"{current_time}.jpg"))
        image = self.visualize_plate(image, coordinates, plate_number) 
        self.save_image(image, os.path.join(self.visualize_review_char_path, f"{current_time}.jpg"))
        
    def visualize_plate(self, image, coordinates, plate_number):
        x1, y1, x2, y2 = coordinates[:4]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        image = cv2.putText(image, f"{plate_number}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image
                         
    def save_image(self, image, path):
        try:
            cv2.imwrite(path, image)
            self.logger.info(f"Image saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save image {path}: {str(e)}")
        
    def crop_plate(self, image, coordinates):
        x1, y1, x2, y2 = coordinates[:4]
        return image[y1:y2, x1:x2]
            
        
        
def test_lpr_data_handler():
    '''
    測試1:兩張圖片，一張有框到車牌，一張沒有框到車牌(完成)
    測試2:兩張圖片，兩張有框到車牌，但是車牌號碼不一樣(完成)
    測試3:兩張圖片，兩張有框到車牌，車牌號碼一樣，信心度夠(完成)
    測試4:兩張圖片，兩張有框到車牌，車牌號碼一樣，但是信心度不夠(完成)
    測試5:兩張圖片，都只有框的結果，但是沒有字符(完成)
    測試5:兩張圖片，字符長度不一樣(完成)
    測試6:兩張圖片，什麼都沒有(完成)

    '''
    lprdatahandler = LPRDataHandler('./continuous_learning_dataset', 0.7)
    
    image1 = cv2.imread('tmp.jpg')
    image2 = cv2.imread('tmp1.jpg')
    images = [image1, image2]

    plate_info = {
        "0": [
            {"coordinates": [736, 104, 885, 197, 0.9297, 0], 
             "plate_number": "3H7797", 
             "carplate_confs": [0.95, 0.96, 0.97, 0.98, 0.99, 0.95], 
             "carplate_coords": [[9, 30, 29, 70], [29, 30, 52, 71], [57, 31, 78, 73], [79, 32, 100, 74], [100, 35, 123, 75], [122, 34, 144, 77]]}
        ],
        "1": [
            {"coordinates": [736, 104, 885, 197, 0.9297, 0], 
             "plate_number": "3H7797", 
             "carplate_confs": [0.85, 0.82, 0.81, 0.84, 0.80, 0.83], 
             "carplate_coords": [[9, 30, 29, 70], [29, 30, 52, 71], [57, 31, 78, 73], [79, 32, 100, 74], [100, 35, 123, 75], [122, 34, 144, 77]]}
        ]
    }

    lprdatahandler.handle_plate_data(plate_info, images)