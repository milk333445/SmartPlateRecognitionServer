import logging
import yaml
import os
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from lprutils import Logger
import signal
import shutil
import glob
import socket
import threading
import subprocess
import time

from test_lprclient import TEST_UDPClient

class ContinuousLearningServer:
    def __init__(self, config_path='./configs/lprserver_config.yaml', logger=None):
        """
        Initialize the ContinuousLearningServer object.

        Args:
            config_path (str): Path to the configuration file (default: './configs/lprserver_config.yaml').
            logger (Logger): Logger object for logging (default: None).
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.settings = config['LPR_UDPServer']
        self.logger = logger if logger else logging.getLogger(__name__)
        self.continuous_learning_setting = self.settings['continuous_learning']
        self.host = self.settings['host']  
        self.reload_port = self.continuous_learning_setting['reload_port']
        self.commands_port = self.continuous_learning_setting['commands_port']
        self.initialize_directories()
        self.train_data_path = self.continuous_learning_setting['train_data_path']
        
        self.images_threshold = self.continuous_learning_setting.get('images_threshold', 100)
        self.continuous_learning_enable = self.continuous_learning_setting['enable']
        self.check_interval_seconds = self.continuous_learning_setting.get('check_interval_hours', 1) * 3600
        self.training_windows = self.continuous_learning_setting['training_windows']
        self.cooldown_period_seconds = self.continuous_learning_setting.get('cooldown_period_hours', 2) * 3600
        self.last_training_time = None
        self.scheduler = BackgroundScheduler()
        self.training = False
        threading.Thread(target=self.listen_for_commands).start()
        self.testclient = None
        
    def initialize_directories(self):
        try:
            self.save_path = self.continuous_learning_setting['save_path']
            self.dataset_path = os.path.join(self.save_path, "dataset")
            self.review_data_char_path = os.path.join(self.dataset_path, "review_data_char")
            self.review_data_path = os.path.join(self.dataset_path, "review_data")
            self.train_data_path = os.path.join(self.dataset_path, "train_data")
            self.image_folder = os.path.join(self.train_data_path, "images")
            self.label_folder = os.path.join(self.train_data_path, "labels")
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
            os.makedirs(self.image_folder, exist_ok=True)
            os.makedirs(self.label_folder, exist_ok=True)
            self.logger.info("All directories initialized successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while initializing directories: {e}")
        
    def listen_for_commands(self):
            """
            Listens for commands on a specified host and port using UDP socket.
            Executes different actions based on the received command.

            Available commands:
            - pause_training: Pauses the training process.
            - resume_training: Resumes the training process, moves model weights, and sends a reload signal to the LPR server.
            - start_testing: Starts testing the model's performance, sends a signal to the LPR server to load the model based on the signal.

            Returns:
            None
            """
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.host, self.commands_port))
                self.logger.info(f"Listening for commands on {self.host}:{self.commands_port}")
                while True:
                    data, addr = sock.recvfrom(1024)
                    command = data.decode()
                    self.logger.info(f"Received command: {command} from {addr}")
                    if command == 'pause_training':
                        self.pause_training()
                    elif command == 'resume_training':
                        self.resume_training()
                        self.move_model_weights()
                        # sent reload signal to the LPR server
                        self.send_signal('reload')
                    elif command == 'start_testing':
                        previous_accuracy = self.extract_accuracy(self.continuous_learning_setting['test_model_results_path'])
                        signal = self.test_model_performance(previous_accuracy)
                        # send signal to the LPR server to load model based on signal
                        self.send_signal(signal)
                    
    def extract_accuracy(self, filename):
        results_path = os.path.join(filename, 'results.txt')
        try:
            with open(results_path, 'r') as file:
                for line in file:
                    if line.startswith('Accuracy'):
                        accuracy = float(line.split('|')[0].split(':')[1].replace('%', '').strip())/100
                        self.logger.info(f"Accuracy extracted from results file: {accuracy}")
                        return accuracy
        except FileNotFoundError:
            self.logger.error(f"Results file not found at path {results_path}")
            return None
        except Exception as e:
            self.logger.error(f"An error occurred while extracting accuracy: {e}")
            return None
        
    def test_model_performance(self, previous_accuracy=None):
        """
        Test the performance of the model by comparing the accuracy with the previous accuracy.

        Args:
            previous_accuracy (float): The accuracy of the previous model. Defaults to None.

        Returns:
            str: 'improved' if the model performance improved, 'not_improved' if the model performance did not improve,
                 None if the previous accuracy is not available.
        """
        if previous_accuracy is not None:
            if self.testclient is None:
                self.testclient = TEST_UDPClient(logger=self.logger)
            self.testclient.send_folder(self.continuous_learning_setting['test_dataset_path'], 'pr')
            self.testclient.close()
            accuracy = self.testclient.accuracy
            if accuracy is not None:
                self.logger.info(f'Previous accuracy: {previous_accuracy}')
                self.logger.info(f"Accuracy of the model now: {accuracy}")
                if accuracy > previous_accuracy:
                    self.logger.info("Model performance improved.")
                    # replace the previous model with the new model
                    return 'improved'
                else:
                    self.logger.info("Model performance did not improve. need to reload the previous model.")
                    return 'not_improved'
        else:
            self.logger.info("Previous accuracy is not available. Cannot compare with the new model.")
            return None
              
    def resume_training(self):
        """
        Resumes the training process if the server is not currently training and is within the training window.

        If the server meets the conditions for resuming training, it searches for the latest run directory and
        retrieves the checkpoint weights. If the checkpoint weights are not found, it resumes training from the
        original pretrain model weights. The training process is then started using the specified command line
        arguments. Once the training process is completed, the server enters the cooldown period.

        If the server is already training or is not within the training window, no training will be done.

        Returns:
            None
        """
        if not self.training and self.is_within_training_window():
            self.logger.info("Resuming training...")
            try:
                latest_run_dir = max(glob.glob(os.path.join(self.continuous_learning_setting['train_file_path'], 'runs', 'train', 'exp*')), key=os.path.getctime)
                weights_folder = os.path.join(latest_run_dir, 'weights')
                if os.path.exists(os.path.join(weights_folder, 'last.pt')):
                    self.logger.info(f"No checkpoints weight, Resuming training from {checkpoint}")
                    checkpoint = os.path.join(weights_folder, 'last.pt')
                else:
                    self.logger.info(f"Resuming training from original pretrain model weights.")
                    checkpoint = self.continuous_learning_setting["pretrain_model_weights_path"]   
                train_command = [
                    'python3', self.continuous_learning_setting["train_script_path"],
                    '--weights', checkpoint,
                    '--cfg', self.continuous_learning_setting["train_model_configs_path"],
                    '--data', self.continuous_learning_setting["train_data_configs_path"],
                    '--hyp', self.continuous_learning_setting["train_hyp_configs_path"],
                    '--epochs', str(self.continuous_learning_setting["train_epochs"]),
                    '--batch-size', str(self.continuous_learning_setting["train_batch_size"]),
                    '--imgsz', str(self.continuous_learning_setting["train_img_size"]),
                    '--freeze', str(self.continuous_learning_setting["freeze_layers"])
                ]
                self.training_process = subprocess.Popen(train_command)
                self.training_process.wait()
                self.logger.info("Training resumed.")
            finally:
                self.training = False
                self.last_training_time = datetime.now()
                self.logger.info("Training completed. Cooldown period started.")
        else:
            self.logger.info("Not within training window or already training. No training will be done.")
        
    def pause_training(self):
        if self.training and self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            time.sleep(2)
            if self.training_process.poll() is None:  
                self.logger.info("Process not terminated, trying kill...")
                self.training_process.kill()
            self.training = False
            self.logger.info("Training paused.")
    
    def start(self):
        if self.continuous_learning_enable:
            self.scheduler.add_job(self.check_images, 'interval', seconds=self.check_interval_seconds)
            self.scheduler.start()
            self.logger.info("Continuous learning server started.")
        else:
            self.logger.info("Continuous learning is disabled.")
        
    def check_images(self):
        if self.last_training_time and (datetime.now() - self.last_training_time).total_seconds() < self.cooldown_period_seconds:
            self.logger.info("Currently in cooldown period. No training will be initiated.")
            return
        
        num_images = len(os.listdir(self.image_folder))
        self.logger.info(f"Number of images in folder: {num_images}")
        
        if num_images >= self.images_threshold and not self.training and self.is_within_training_window():
            self.logger.info("Starting training...")
            # send delete LPRData signal to the LPR server
            self.send_signal('delete')
            self.move_file_to_training_folder()
            self.initiate_training()
            self.move_model_weights()
            # sent reload signal to the LPR server
            self.send_signal('reload')
            
        elif not self.is_within_training_window():
            self.logger.info("Not within training window. No training will be done.")
            
    def move_file_to_training_folder(self):
        try:
            dest_image_folder = os.path.join(self.train_data_path, 'images')
            dest_label_folder = os.path.join(self.train_data_path, 'labels')
            
            os.makedirs(dest_image_folder, exist_ok=True)
            os.makedirs(dest_label_folder, exist_ok=True)

            self.copy_files(self.image_folder, dest_image_folder)

            self.copy_files(self.label_folder, dest_label_folder)
            
        except Exception as e:
            self.logger.error(f"Error merging folders: {str(e)}")
            
    def copy_files(self, src_folder, dest_folder):
        for filename in os.listdir(src_folder):
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            if not os.path.exists(dest_path):  
                shutil.copy(src_path, dest_path)
                self.logger.info(f"Copied {src_path} to {dest_path}")
            else:
                self.logger.info(f"File {dest_path} already exists. Skipping.")
            
    def is_within_training_window(self):
        current_time = datetime.now().time()
        for window in self.training_windows:
            start_time = datetime.strptime(window['start'], '%H:%M').time()
            end_time = datetime.strptime(window['end'], '%H:%M').time()
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    return True
            else: # cross midnight
                if start_time <= current_time or current_time <= end_time:
                    return True
        return False
            
    def initiate_training(self):
        """
        Initiates the training process for the continuous learning server.

        This method performs the following steps:
        1. Sets the training flag to True.
        2. Splits the dataset using the 'split_train_val_data.py' script.
        3. Executes the training script with the specified parameters.
        4. Waits for the training process to complete.
        5. Sets the training flag to False.
        6. Records the completion time and logs a message.

        Note: The training process is executed as a subprocess.

        Returns:
            None
        """
        try:
            self.training = True
            self.training_process = None
            # split the dataset
            split_dataset_script = os.path.join(self.train_data_path, 'split_train_val_data.py')
            split_command = [
                'python3', split_dataset_script,
                '--train_ratio', str(self.continuous_learning_setting["train_ratio"])
            ]
            subprocess.run(split_command)
            # train
            train_command = [
                'python3', self.continuous_learning_setting["train_script_path"],
                '--weights', self.continuous_learning_setting["pretrain_model_weights_path"],
                '--cfg', self.continuous_learning_setting["train_model_configs_path"],
                '--data', self.continuous_learning_setting["train_data_configs_path"],
                '--hyp', self.continuous_learning_setting["train_hyp_configs_path"],
                '--epochs', str(self.continuous_learning_setting["train_epochs"]),
                '--batch-size', str(self.continuous_learning_setting["train_batch_size"]),
                '--imgsz', str(self.continuous_learning_setting["train_img_size"]),
                '--freeze', str(self.continuous_learning_setting["freeze_layers"])
            ]
            
            self.training_process = subprocess.Popen(train_command)
            self.training_process.wait()
                
        finally:
            self.training = False
            self.last_training_time = datetime.now()
            self.logger.info("Training completed. Cooldown period started.")
        
    def send_signal(self, message='reload'):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            message = message
            sock.sendto(message.encode(), (self.host, self.reload_port))
            self.logger.info(f"Sent {message} signal to {self.host}:{self.reload_port}")
       
    def move_model_weights(self):
        latest_run_dir = max(glob.glob(os.path.join(self.continuous_learning_setting['train_file_path'], 'runs', 'train', 'exp*')), key=os.path.getctime)
        weights_folder = os.path.join(latest_run_dir, 'weights')
        self.logger.info(f'Latest run dir: {latest_run_dir}')
        destination_folder = self.continuous_learning_setting['tmp_train_model_weights_path']
        os.makedirs(destination_folder, exist_ok=True)
        
        for filename in ['best.pt', 'last.pt']:
            src_path = os.path.join(weights_folder, filename)
            dest_path = os.path.join(destination_folder, filename)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                self.logger.info(f"Copied weight {src_path} to {dest_path}.")
            else:
                self.logger.error(f"Weight file {src_path} does not exist.")
        
    def shutdown(self):
        self.scheduler.shutdown()
        self.logger.info("Continuous Learning Server shut down.")
        
def main():
    logger = Logger('LPR_ContinuousLearningServer', './logs/lpr_continuousLearning_server.log', level=logging.DEBUG).get_logger()
    continuous_learning_server  = ContinuousLearningServer(logger=logger)
    def signal_handler(signal, frame):
        logger.info("Received signal to stop server.")
        continuous_learning_server.shutdown()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    continuous_learning_server.start()
    signal.pause()
        
if __name__ == "__main__":
    main()