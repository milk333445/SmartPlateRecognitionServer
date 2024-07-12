import json
import time
import numpy as np
import base64
import logging
from lprutils import Logger, display_plates
from collections import defaultdict
import argparse


class TEST_UDPClient:
    def __init__(self, config_path='./configs/test_lprclient_config.yaml', logger=None):
        """
        Initializes an instance of the LPRClient class.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to './configs/test_lprclient_config.yaml'.
            logger (Logger, optional): Logger object for logging messages. Defaults to None.
        """
        self.load_config(config_path)
        self.results_file = os.path.join(self.save_path, 'results.txt')
        self.output_folder = os.path.join(self.save_path, 'output')
        os.makedirs(self.output_folder, exist_ok=True)
        self.responses = {}
        self.correct_predictions = 0
        self.total_count_per_characters = defaultdict(int)
        self.correct_count_per_characters = defaultdict(int)
        self.misidentifications = []
        self.length_mismatches = []
        self.undetected_images = []
        self.char_classes = {i: chr(65+i) if i < 26 else str(i-26) for i in range(36)}
        self.received_responses = 0
        self.failed_transmissions = 0
        self.sent_images = 0
        self.logger = logger if logger else logging.getLogger(__name__)
        
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Config file not found at path {config_path}")
            raise
            
        settings = config['TEST_LPR_UDPClient']
        self.server_host = settings['host']
        self.server_port = settings['port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(settings['timeout'])
        self.draw_bbox = settings['draw']
        self.send_mode = settings.get('send_mode', 'image')
        self.send_interval = settings.get('send_interval', 0)
        self.save_path = settings['save_path']
        
        
    def send_folder(self, folder_path, mode):
        start_time = time.time()
        self.load_images(folder_path)
        for image_path in self.images:
            self.sent_images += 1
            timestamp = time.strftime("%y%m%d%H%M%S%f")[:-3]
            self.send(image_path, mode, timestamp)
            time.sleep(self.send_interval)  # 控制發送頻率
            response = self.receive_response(image_path)
            if response:
                self.responses[image_path] = response
                self.received_responses += 1
            else:
                self.failed_transmissions += 1
                self.logger.error(f"Failed to receive response for image {image_path}")
        end_time = time.time()
        self.calculate_statistics()
        self.logger.info('-'*50)
        self.logger.info(f"Processed {len(self.images)} images in {end_time - start_time:.4f} seconds")
        self.logger.info(f"Average processing time per image: {(end_time - start_time) / len(self.images):.4f} seconds")
        self.logger.info(f"Received responses for {self.received_responses} out of {self.sent_images} sent images")
        self.logger.info(f"Failed transmissions: {self.failed_transmissions}")
        
    def load_images(self, folder_path):
        self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        length = len(self.images)
        self.logger.info(f"Found {length} images in folder {folder_path}")
        return self.images
        
    def calculate_statistics(self):
        """
        Calculates statistics for the license plate recognition results.

        This method iterates over the responses dictionary, which contains the image paths and corresponding
        recognition responses. For each image, it compares the detected license plate with the true license
        plate and updates various statistics accordingly.

        The statistics calculated include:
        - Number of correct predictions
        - Total count of characters per license plate
        - Correct count of characters per license plate
        - Misidentifications (character-level differences between detected and true license plates)
        - Length mismatches (differences in the number of characters between detected and true license plates)
        - Undetected images (images for which no license plate was detected)

        After calculating the statistics, it logs the results using the `log_statistics` method.

        Note: This method assumes that the `responses` dictionary is populated with the necessary data.

        Returns:
            None
        """
        for image_path, response in self.responses.items():
            image_name = os.path.basename(image_path)
            True_car_plate, _ = os.path.splitext(image_name)
            detection = response.get('0', [])
            if detection:
                detected_plate = detection[0].get('plate_number', '')
                if detected_plate == True_car_plate:
                    self.correct_predictions += 1
                if len(detected_plate) == len(True_car_plate):
                    for i in range(len(detected_plate)):
                        self.total_count_per_characters[True_car_plate[i]] += 1
                        if detected_plate[i] == True_car_plate[i]:
                            self.correct_count_per_characters[True_car_plate[i]] += 1
                        else:
                            self.misidentifications.append((image_path, i, True_car_plate[i], detected_plate[i]))
                else:
                    self.length_mismatches.append((image_path, True_car_plate, detected_plate))
            else:
                self.undetected_images.append(image_path)
        self.log_statistics()
    
    def log_statistics(self):
        with open(self.results_file, 'w') as file:
            self.accuracy = self.correct_predictions / len(self.responses) if len(self.responses) > 0 else 0
            file.write(f"Accuracy: {self.accuracy * 100:.2f}% | {self.correct_predictions}/{len(self.responses)}\n")
            self.logger.info(f"accuracy: {self.accuracy * 100:.2f}% | {self.correct_predictions}/{len(self.responses)}")  
            self.log_character_accuracy(file)
            file.write(f"Received responses for {self.received_responses} out of {self.sent_images} sent images\n")
            file.write(f"Failed transmissions: {self.failed_transmissions}\n")
            self.logger.info("Statistics written to file.")
            
    def log_character_accuracy(self, file):
        for index in self.char_classes:
            char = str(self.char_classes[index])
            total = self.total_count_per_characters[char]
            correct = self.correct_count_per_characters[char]
            accuracy = correct / total if total > 0 else 0
            if total > 0:
                self.logger.info(f"{char}: {accuracy * 100:.2f}% | {correct}/{total}")
                file.write(f"{char}: {accuracy * 100:.2f}% | {correct}/{total}\n")
            else:
                self.logger.info(f"{char}: No data")
                file.write(f"{char}: No data\n")
                
        for misid in self.misidentifications:
            img_file, char_index, true_char, detected_char = misid
            self.logger.info(f"In image {img_file}, character {char_index} is misidentified as {detected_char} instead of {true_char}")
            file.write(f"In image {img_file}, character {char_index} is misidentified as {detected_char} instead of {true_char}\n")
        for mismatch in self.length_mismatches:
            img_file, true_plate, detected_plate = mismatch
            self.logger.info(f"In image {img_file}, the detected plate {detected_plate} has a different length than the true plate {true_plate}")
            file.write(f"In image {img_file}, the detected plate {detected_plate} has a different length than the true plate {true_plate}\n")
        for img in self.undetected_images:
            self.logger.info(f"Image {img} was not detected")
            file.write(f"Image {img} was not detected\n")
                             
    def send(self, image_path, mode, timestamp, chunk_size=1024):
        images_data = {}
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                if self.send_mode == 'path':
                    images_data['image_0'] = image_path
                else:
                    img = cv2.imread(image_path)
                    _, buffer = cv2.imencode('.jpg', img)
                    img_data = base64.b64encode(buffer).decode('utf-8')
                    images_data['image_0'] = img_data
                    
                packet = json.dumps({
                    'mode': mode,
                    'timestamp': timestamp,
                    'images': images_data,
                    'image mode': self.send_mode
                })
            
                # 計算要傳的總包數
                total_packets = (len(packet) + chunk_size - 1) // chunk_size
                for i in range(0, len(packet), chunk_size):
                    chunk_data = packet[i:i+chunk_size].encode()
                    header = f"{i//chunk_size}/{total_packets}".encode().ljust(32) # 統一長度為32bytes
                    self.sock.sendto(header + chunk_data, (self.server_host, self.server_port))
                    
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
    
    def receive_response(self, image_path):
        try:
            self.logger.info("Waiting for response from server...")
            data, addr = self.sock.recvfrom(1024)
            response_data = json.loads(data.decode())
            if self.draw_bbox:
                for image_index, plates_info in response_data.items():
                    img = cv2.imread(image_path)
                    img = display_plates(img, plates_info, (1000, 600))
                    save_path = os.path.join(self.output_folder, f"annotated_{os.path.basename(image_path)}")
                    cv2.imwrite(save_path, img)
            self.logger.info('Response received from server')
            self.logger.debug(f"Received response from server: {data.decode()}")
            return response_data
        except socket.timeout:
            self.logger.error("Timeout: Server is not responding.")
            return None
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
    
    def close(self):
        self.sock.close()
        self.logger.info("Socket closed")
        
def main(args):
    logger = Logger('TEST_UDPClient', './logs/test_udp_client.log', level=args.logging_level).get_logger()
    client = TEST_UDPClient(logger=logger)
    
    client.send_folder(args.folder_path, args.mode)
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send images from a folder to TEST server via UDP.')
    parser.add_argument('--folder_path', type=str, default='./dataset/labeled_images',
                        help='Path to the folder containing labeled images.')
    parser.add_argument('--mode', type=str, choices=['pr', 'ppd', 'pgs'], default='pr',
                        help='Mode of operation: pr (plate recognition), ppd (plate presence detection), pgs (other).')
    parser.add_argument('--logging_level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()
    main(args)