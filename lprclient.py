import socket
import base64
import json
from datetime import datetime
from lprutils import Logger, display_plates
import cv2
import numpy as np
import yaml
import logging
import time
import argparse

class LPR_UDPClient:
    def __init__(self, config_path='./configs/lprclient_config.yaml', logger=None):
        """
        Initialize the LPRClient object.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to './configs/lprclient_config.yaml'.
            logger (Logger, optional): Logger object for logging. Defaults to None.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.logger = logger if logger else logging.getLogger(__name__)
        settings = config['LPR_UDPClient']
        self.server_host = settings['host']
        self.server_port = settings['port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(settings['timeout'])
        self.draw_bbox = settings['draw']
        self.send_mode = settings.get('send_mode', 'image') # default send image
        
    def send(self, images, mode, timestamp, chunk_size=1024):
        """
        Sends the images to the server.

        Args:
            images (str or np.ndarray or list): The images to be sent. It can be a single image path (str),
                a single image as a numpy array (np.ndarray), or a list of image paths or numpy arrays.
            mode (str): The mode of the request.
            timestamp (str): The timestamp of the request.
            chunk_size (int, optional): The size of each chunk to be sent. Defaults to 1024.

        Raises:
            ValueError: If an image at the specified path cannot be loaded.
            socket.timeout: If the server does not respond within the specified timeout.
            Exception: If any other error occurs.

        Returns:
            None
        """
        self.images = []
        images_data = {}
        if isinstance(images, (str, np.ndarray)):
            images = [images]
        for index, item  in enumerate(images):
            if isinstance(item, str):
                try:
                    if not item.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.logger.error(f"Unsupported file extension for image {item}")
                        continue
                    img = cv2.imread(item)
                    if img is None:  
                        raise ValueError(f"Image at path {item} could not be loaded.")
                    self.images.append(item)
                except Exception as e:
                    self.logger.error(f"An error occurred while reading image {item}: {e}")
                    continue
            elif isinstance(item, np.ndarray):
                img = item
                self.images.append(img)
            else:
                self.logger.error("Unsupported image format")
                continue
            
            if self.send_mode == 'path':
                images_data[f'image_{index}'] = item
            else:
                _, buffer = cv2.imencode('.jpg', img)
                img_data = base64.b64encode(buffer).decode('utf-8')
                images_data[f'image_{index}'] = img_data

        try:
            packet = json.dumps(
                {
                    'mode': mode,
                    'timestamp': timestamp,
                    'images': images_data,
                    'image mode': self.send_mode
                }
            )
            
            # Calculate the total number of packets to be sent
            total_packets = (len(packet) + chunk_size - 1) // chunk_size
            for i in range(0, len(packet), chunk_size):
                chunk_data = packet[i:i+chunk_size].encode()
                header = f"{i//chunk_size}/{total_packets}".encode().ljust(32) # Fixed length of 32 bytes
                self.sock.sendto(header + chunk_data, (self.server_host, self.server_port))
            
            self.receive_response()
            
        except socket.timeout:
            self.logger.error("Timeout: Server is not responding.")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
    
    def receive_response(self):
        try:
            self.logger.info("Waiting for response from server...")
            data, addr = self.sock.recvfrom(1024)
            response_data = json.loads(data.decode())
            if self.draw_bbox:
                for image_index, plates_info in response_data.items():
                    img = self.images[int(image_index)]
                    if img is None:
                        self.logger.error(f"Failed to load image from img{int(image_index)}")
                        continue
                    img = display_plates(img, plates_info, (1000, 600))
                    cv2.imshow(f"Image {image_index}", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            self.logger.info('Response received from server')
            self.logger.debug(f"Received response from server: {data.decode()}")
        except socket.timeout:
            self.logger.error("Timeout: Server is not responding.")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
    
    def close(self):
        self.sock.close()
        
def main(args):
    logger = Logger('LPR_UDPClient', './logs/lpr_udp_client.log', level=args.logging_level).get_logger()
    client = LPR_UDPClient(config_path='./configs/lprclient_config.yaml', logger=logger)
    
    images_path = args.image_paths
    images = [cv2.imread(path) for path in images_path]  # Load all images from the paths provided
    
    timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
    upd_send_start_time = time.time()
    
    client.send(images_path, args.mode, timestamp)
    print('UDP send time:', time.time() - upd_send_start_time)
    client.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send images to LPR server via UDP.')
    parser.add_argument('image_paths', type=str, nargs='+',
                        help='One or more paths to images to be processed.')
    parser.add_argument('--mode', type=str, choices=['pr', 'ppd', 'pgs'], default='pr',
                        help='Mode of operation: pr (plate recognition), ppd (plate presence detection), pgs (other).')
    parser.add_argument('--logging_level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()
    main(args)