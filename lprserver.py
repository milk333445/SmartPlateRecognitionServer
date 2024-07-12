import socket
import sys
import numpy as np
import json
import base64
from PIL import Image
import io
import cv2
import time
from lprutils import Logger, load_image
from lpr_engine.lpr_predictor import LPRDetector
import threading
import logging
import yaml
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import shutil
import argparse

from lprdatahandler import LPRDataHandler

class LPR_UDPServer:
    """
    A UDP server for License Plate Recognition (LPR).

    Args:
        config_path (str): Path to the configuration file (default: './configs/lprserver_config.yaml').
        detect_config_path (str): Path to the detection configuration file (default: './lpr_engine/configs/detection.yaml').
        logger (Logger): Logger object for logging messages (default: None).

    Attributes:
        detect_config_path (str): Path to the detection configuration file.
        settings (dict): Settings loaded from the configuration file.
        host (str): Host address to bind the server.
        port (int): Port number to bind the server.
        sock (socket.socket): UDP socket object.
        buffer_timeout (int): Timeout value for receiving packets.
        packet_data (dict): Dictionary to store received packet data.
        packet_receipt_time (dict): Dictionary to store the receipt time of packets.
        total_packets_expected (dict): Dictionary to store the total number of expected packets.
        packet_count (dict): Dictionary to store the count of received packets.
        time_status (dict): Dictionary to store the status of timeout occurrence.
        logger (Logger): Logger object for logging messages.
        LPRdetector (LPRDetector): LPRDetector object for license plate detection.
        data_lock (threading.Lock): Lock object for thread synchronization.
        continuous_learning_setting (dict): Continuous learning settings.
        datahandler (LPRDataHandler): LPRDataHandler object for continuous learning data collection.
        training (bool): Flag indicating whether the server is in training mode.

    Methods:
        set_buffer_sizes(recv_size, send_size): Set the buffer sizes for the socket.
        listen(): Start listening for incoming packets.
        send_command_to_CLServer(command): Send a command to the Continuous Learning Server.
        listen_for_cls_signal(): Listen for reload signals from the Continuous Learning Server.
        delete_LPRDetector(): Delete the LPRDetector object.
        replace_model(src, dest): Replace the model files with the ones from the source path.
        extract_model_files(config_path, key): Extract the model files path from the configuration file.
        reload_model(detection_config_path): Reload the LPRDetector model.
        init_connection(addr, total): Initialize a connection with a client.
        update_packet_data(addr, sequence, data): Update the packet data received from a client.
        handle_complete_data(addr): Handle the complete data received from a client.
        process_and_cleanup(addr): Process the complete data and clean up resources.
        process_packet(data, addr): Process a packet of data received from a client.
        process_data(json_data): Process the JSON data received from a client.
    """
    def __init__(self, config_path='./configs/lprserver_config.yaml', detect_config_path='./lpr_engine/configs/detection.yaml', logger=None):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.detect_config_path = detect_config_path
        self.settings = config['LPR_UDPServer']
        self.host = self.settings['host']  
        self.port = self.settings['port']  
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
        self.sock.bind((self.host, self.port))  
        self.buffer_timeout = self.settings['buffer_timeout']
        self.packet_data = {}
        self.packet_receipt_time = {}
        self.total_packets_expected  = {}
        self.packet_count = {}
        self.time_status = {}
        self.logger = logger if logger else logging.getLogger(__name__)
        self.LPRdetector = LPRDetector(self.detect_config_path, self.settings['perspect_transform'])
        self.set_buffer_sizes(self.settings['recv_buffer_size'], self.settings['send_buffer_size'])
        
        self.data_lock = threading.Lock()
        self.start_timeout_checker()
        
        # continuous learning data collection
        self.continuous_learning_setting = self.settings['continuous_learning']
        if self.continuous_learning_setting['enable']:
            # 初始化資料蒐集器class
            self.logger.info("Continuous learning data collection is enabled.")
            self.datahandler = LPRDataHandler(self.continuous_learning_setting['save_path'], self.continuous_learning_setting['conf_threshold'], logger=self.logger)
            threading.Thread(target=self.listen_for_cls_signal).start()
            self.training = False  
    def set_buffer_sizes(self, recv_size, send_size):
        """
        Set the buffer sizes for receiving and sending data over the socket.

        Args:
            recv_size (int): The size of the receive buffer in bytes.
            send_size (int): The size of the send buffer in bytes.

        Returns:
            None

        Raises:
            None
        """
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_size)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_size)
        self.logger.info(f"Buffer sizes set to: recv={recv_size} bytes, send={send_size} bytes")

    def listen(self):
        """
        Listens for incoming data packets and handles them accordingly.

        This method continuously listens for incoming data packets on the server's socket.
        It receives the data, extracts the header information, and updates the packet data accordingly.
        If all packets for a particular address have been received, it calls the `handle_complete_data` method.

        Raises:
            KeyboardInterrupt: If the server is interrupted by a keyboard interrupt (e.g., Ctrl+C).

        """
        self.logger.info(f"Server is listening on {self.host}:{self.port}")
        try:
            while True:
                data, addr = self.sock.recvfrom(1056) # 1024 bytes for data, 32 bytes for header
                header = data[:32].decode().strip() # ex: 0/3
                sequence, total = map(int, header.split('/')) # ex: 0, 3
                if addr not in self.packet_data:
                    self.init_connection(addr, total)
                self.update_packet_data(addr, sequence, data[32:])
                if self.packet_count[addr]['received'] == total:
                    self.logger.info(self.packet_count)
                    self.handle_complete_data(addr)
                
        except KeyboardInterrupt:
            self.logger.info("Server is shutting down.")
            self.shutdown()
            
    def send_command_to_CLServer(self, command):
        cls_server_port = self.continuous_learning_setting['commands_port']
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(command.encode(), (self.host, cls_server_port))
            self.logger.info(f"Command sent to CLServer: {command}")
            
    def listen_for_cls_signal(self):
        """
        Listens for continuous learning signals and performs corresponding actions based on the received signal.

        This method binds a socket to the specified host and port to listen for continuous learning signals.
        When a signal is received, it performs the appropriate action based on the signal value.

        Signals:
        - 'reload': Reloads the model and tests its performance.
        - 'delete': Deletes the LPRDetector.
        - 'improved': Indicates that the model performance has improved and reloads the model.
        - 'not_improved': Indicates that the model performance did not improve and reloads the previous model.

        Returns:
        None
        """
        self.reload_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reload_socket.bind((self.host, self.continuous_learning_setting['reload_port']))
        self.logger.info(f"Listening for reload signal on {self.host}:{self.continuous_learning_setting['reload_port']}")
        try:
            while True:
                data, addr = self.reload_socket.recvfrom(1024)
                if data.decode() == 'reload':
                    self.reload_model(self.continuous_learning_setting['tmp_detection_config_path'])
                    # Test model performance
                    self.logger.info("Testing model performance after reload...")
                    # Send 'start_testing' command to CLServer
                    self.send_command_to_CLServer('start_testing')

                elif data.decode() == 'delete':
                    self.logger.info("Deleting LPRDetector...")
                    self.delete_LPRDetector()

                elif data.decode() == 'improved':
                    self.logger.info("Model performance improved.")
                    # Reload the model
                    self.replace_model(self.continuous_learning_setting['tmp_detection_config_path'], self.detect_config_path)

                elif data.decode() == 'not_improved':
                    self.logger.info("Model performance did not improve.")
                    # Reload the previous model
                    self.reload_model(self.detect_config_path)

        finally:
            self.reload_socket.close()
                
    def delete_LPRDetector(self):
        # delete LPRDetector(free up memory)
        if hasattr(self, 'LPRdetector'):
            del self.LPRdetector
            self.training = True
            self.logger.info("LPRDetector deleted and resources are freed.")
                
    def replace_model(self, src, dest):
        """
        Replaces the model files from the source path to the destination path.

        Args:
            src (str): The source path of the model files.
            dest (str): The destination path where the model files will be replaced.

        Returns:
            None

        Raises:
            FileNotFoundError: If either the source or destination model files path does not exist.
            Exception: If there is an error while replacing the model files.

        """
        # extract the model files path
        src_model_path = self.extract_model_files(src, key='car_char')
        dest_model_path = self.extract_model_files(dest, key='car_char')
        
        if os.path.exists(dest_model_path) and os.path.exists(src_model_path):
            os.remove(dest_model_path)
            self.logger.info("Replacing model files...")
            try:
                shutil.copy(src_model_path, dest_model_path)
                self.logger.info("Model files replaced.")
            except Exception as e:
                self.logger.error(f"Error replacing model files: {e}")

    def extract_model_files(self, config_path, key):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        model_files = config[key]['weight']
        return model_files
    
    def reload_model(self, detection_config_path=None):
        self.logger.info("Reloading model...")
        self.LPRdetector = LPRDetector(detection_config_path, self.settings['perspect_transform'])
        self.logger.info("Model reloaded.")
        
    def init_connection(self, addr, total):
        """
        Initializes a connection with the given address and total number of packets expected.

        Args:
            addr (str): The address of the connection.
            total (int): The total number of packets expected.

        Returns:
            None

        """
        self.packet_data[addr] = {}
        self.packet_receipt_time[addr] = time.time()
        self.total_packets_expected[addr] = total
        self.packet_count[addr] = {'received': 0, 'total': total}
        self.time_status[addr] = False
        self.logger.debug(f"Initialized connection from {addr} expecting {total} packets.")
        
    def update_packet_data(self, addr, sequence, data):
        self.packet_data[addr][sequence] = data
        self.packet_count[addr]['received'] += 1
            
    def handle_complete_data(self, addr):
        """
        Handles complete data received from a specific address.

        Args:
            addr (str): The address from which the data was received.

        Returns:
            None

        Raises:
            None
        """
        # check if LPRDetector is deleted
        if not hasattr(self, 'LPRdetector'):
            # tell cls server to pause the training
            self.send_command_to_CLServer('pause_training')
            self.LPRdetector = LPRDetector(self.detect_config_path, self.settings['perspect_transform'])
        if not self.time_status.get(addr, False):
            self.logger.info(f"All packets received from {addr}. Processing data...")
            self.process_and_cleanup(addr)
        else:
            self.logger.info(f"Timeout occurred from {addr}. No further processing will be done.")
         
    def process_and_cleanup(self, addr):
        full_data = b''.join(self.packet_data[addr][i] for i in sorted(self.packet_data[addr], key=int))
        self.process_packet(full_data, addr)

    def process_packet(self, data, addr):
        """
        Process a packet of data received from a client.

        Args:
            data (str): The data received from the client.
            addr (tuple): The address of the client.

        Raises:
            json.JSONDecodeError: If there is an error decoding the JSON data.
            Exception: If there is an error processing the packet.

        Returns:
            None
        """
        try:
            json_data = json.loads(data)
            detection_mode = json_data['mode']
            images = self.process_data(json_data)
            plates_info = self.LPRdetector.process_images(images, mode=detection_mode)
            response_data = json.dumps(plates_info).encode('utf-8')
            self.logger.debug(f'Received packet {self.packet_count[addr]["received"]}/{self.packet_count[addr]["total"]}')
            self.logger.info(f"Received and saved image. Mode: {json_data['mode']}, Timestamp: {json_data['timestamp']}")
            self.respond(response_data, addr)

            if self.continuous_learning_setting['enable'] and detection_mode == 'pr':
                self.datahandler.handle_plate_data(plates_info, images)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error while processing the packet from {addr}: {e}")
            self.respond(b"Error processing your data.", addr)

        except Exception as e:
            self.logger.error(f"An error occurred while processing the packet: {e}")
            self.respond(b"Error processing your image.", addr)

        finally:
            if addr in self.packet_data:
                del self.packet_data[addr]
            if addr in self.packet_receipt_time:
                del self.packet_receipt_time[addr]
            if addr in self.packet_count:
                del self.packet_count[addr]
            if addr in self.total_packets_expected:
                del self.total_packets_expected[addr]
            self.logger.debug(f"Cleaned up resources for {addr}")

            if self.training:
                self.send_command_to_CLServer('resume_training')
                self.training = False
            
    def process_data(self, json_data):
        image_mode = json_data['image mode']
        images = [None] * len(json_data['images'])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(load_image, img_content, image_mode): index for index, (img_key, img_content) in enumerate(json_data['images'].items())}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                if result is not None:
                    images[index] = result
        return images
                    
    def respond(self, data, addr):
        self.sock.sendto(data, addr)

    def shutdown(self):
        if hasattr(self, 'timeout_timer'):
            self.timeout_timer.cancel()
        self.sock.close() 
        self.logger.info("Server is shut down.")
        self.reload_socket.close()
        self.logger.info("Reload socket is shut down.")
        sys.exit(0)  
        
    def start_timeout_checker(self):
        self.timeout_timer = threading.Timer(1, self.check_all_timeouts)
        self.timeout_timer.daemon = True
        self.timeout_timer.start()
    
    def check_all_timeouts(self):
        """
        Checks for packet timeouts and performs necessary actions.

        This method iterates over the packet receipt times and checks if any packets have timed out
        based on the buffer timeout value. If a timeout is detected, it logs a warning, sends a timeout
        response to the corresponding address, and marks the address as timed out. It also cleans up
        resources associated with the timed out address.

        Returns:
            None
        """
        current_time = time.time()
        addresses_to_remove = []
        for addr in list(self.packet_receipt_time.keys()):
            if current_time - self.packet_receipt_time[addr] > self.buffer_timeout:
                self.logger.warning(f"Packet loss detected from {addr}. Timeout triggered.")
                self.logger.debug(f"Received {self.packet_count[addr]['received']}/{self.total_packets_expected[addr]} packets so far.")
                timeout_response = b"Timeout occurred. No further packets will be processed."
                self.respond(timeout_response, addr)
                self.time_status[addr] = True
                addresses_to_remove.append(addr)
        for addr in addresses_to_remove:
            self.cleanup_resources(addr)

        self.start_timeout_checker()
        
    def cleanup_resources(self, addr):
        with self.data_lock:
            if addr in self.packet_data:
                del self.packet_data[addr]
            if addr in self.packet_receipt_time:
                del self.packet_receipt_time[addr]
            if addr in self.packet_count:
                del self.packet_count[addr]
            if addr in self.total_packets_expected:
                del self.total_packets_expected[addr]
        self.logger.debug(f"Cleaned up resources for {addr}")

def main(args):
    logger = Logger('LPR_UDPServer', './logs/lpr_udp_server.log', level=args.logging_level).get_logger()
    server = LPR_UDPServer(config_path=args.config_path, logger=logger)
    server.listen()

if __name__ == "__main__":
    # 重點!
    # sudo sysctl -w net.core.rmem_default=524288
    # sudo sysctl -w net.core.rmem_max=1048576 增加緩衝區可以減少數據丟失的可能性(有用)
    parser = argparse.ArgumentParser(description='Start LPR UDP Server.')
    parser.add_argument('--config_path', type=str, default='./configs/lprserver_config.yaml',
                        help='Path to the server configuration file.')
    parser.add_argument('--logging_level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()
    main(args)
