import os
import sys
import argparse
import glob
import keyboard
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pygame.camera


class Func():
    def __init__(self):
        self.record = ""
        self.camera_capture = None  # Renamed from 'camera' to avoid conflict with pygame.camera
        self.model_path = ""
        self.img_source = ""
        self.min_thresh = 0.5
        self.user_res = ""
        self.t_start = 0
        self.wait_time = 0
        self.resize = False  # Instance attribute for resize flag
        self.resW = 0
        self.resH = 0
        self.usb_idx = None
        self.source_type = ''
        self.vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
        self.img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
        self.labels = {}
        self.model = None
        self.fps_avg_len = 200
        self.frame_rate_buffer = []
        self.avg_frame_rate = 0.0
        self.bbox_colors = []
        self.recorder = None
        self.imgs_list = []
        self.img_count = 0

    def check_key_press(self, key_press: str) -> int:
        print(f"Press the '{key_press}' key to continue...")
        event_key = keyboard.read_event(suppress=True)  # suppress=True prevents the key from being typed
        if event_key.name == key_press and event_key.event_type == keyboard.KEY_DOWN:
            return 1
        else:
            return 0

    def init_parse(self):
        parser = argparse.ArgumentParser(description="YOLO Object Detection with different sources")
        parser.add_argument("--model", help="Path to YOLO model file (example: runs/detect/train/weights/best.pt)", required=True)
        parser.add_argument("--source", help="Image source: image file (test.jpg), folder (test_dir), video file (testvid.mp4), or USB camera (usb0, usb1...)", required=True)
        parser.add_argument("--thresh", type=float, help="Minimum confidence threshold for displaying detected objects (example: 0.4)", default=0.5)
        parser.add_argument("--resolution", help="Resolution in WxH format (example: 640x480). Defaults to source resolution", default=None)
        parser.add_argument("--record", help="Record results from video or webcam as demo1.avi (requires --resolution)", action="store_true")
        args = parser.parse_args()

        self.model_path = args.model
        self.img_source = args.source
        self.min_thresh = float(args.thresh)
        self.user_res = args.resolution
        self.record = args.record

    def check_file_path(self, file_path: str) -> int:
        if not os.path.exists(file_path):
            return 1
        else:
            return 0

    def calculate_fps(self):
        t_stop = time.perf_counter()
        frame_duration = t_stop - self.t_start
        if frame_duration > 0:
            frame_rate_calc = 1.0 / frame_duration

            if len(self.frame_rate_buffer) >= self.fps_avg_len:
                self.frame_rate_buffer.pop(0)
            self.frame_rate_buffer.append(frame_rate_calc)

            self.avg_frame_rate = np.mean(self.frame_rate_buffer)
        else:
            self.avg_frame_rate = 0.0

    def check_folder(self):
        for file in glob.glob(os.path.join(self.img_source, '*')):
            _, file_ext = os.path.splitext(file)
            if file_ext in self.img_ext_list:
                self.imgs_list.append(file)
        if not self.imgs_list:
            print(f"No image files found in the directory: {self.img_source}")
            sys.exit(0)

    def check_img_src(self):
        print(self.img_source)
        if os.path.isdir(self.img_source):
            self.source_type = 'folder'
        elif os.path.isfile(self.img_source):
            ext = os.path.splitext(self.img_source)[1]
            if ext in self.img_ext_list:
                self.source_type = 'image'
            elif ext in self.vid_ext_list:
                self.source_type = 'video'
            else:
                print(f'File extension {ext} is not supported.')
                sys.exit(0)
        elif 'usb' in self.img_source.lower():
            self.source_type = 'usb'
            try:
                self.usb_idx = int(self.img_source[3:])
                # Attempt to open the camera to confirm index
                temp_cam = cv2.VideoCapture(self.usb_idx)
                if not temp_cam.isOpened():
                    print(f"Error: Could not open USB camera with index: {self.usb_idx}. Please ensure it's connected and the index is correct.")
                    print("Attempting to scan for available cameras...")
                    self.scan_usb_cameras()
                    sys.exit(0)
                else:
                    print(f"USB camera opened successfully with index {self.usb_idx}.")
                    temp_cam.release() # Release immediately after check
            except ValueError:
                print(f'Invalid USB camera index: {self.img_source[3:]}. Please use format like "usb0", "usb1".')
                self.scan_usb_cameras()
                sys.exit(0)
            except Exception as e:
                print(f"An unexpected error occurred while checking USB camera: {e}")
                self.scan_usb_cameras()
                sys.exit(0)

        elif 'picamera' in self.img_source.lower():
            self.source_type = 'picamera'
        else:
            print(f'Input {self.img_source} is invalid. Please try again.')
            sys.exit(0)

    def scan_usb_cameras(self):
        print("Scanning for USB cameras...")
        found_cameras = []
        for i in range(10): # Check indices 0 through 9
            cap_check = cv2.VideoCapture(i)
            if cap_check.isOpened():
                found_cameras.append(i)
                cap_check.release()
        if found_cameras:
            print(f"Found camera(s) at index(es): {', '.join(map(str, found_cameras))}")
            print("Please use the correct index in your --source argument (e.g., --source usb0 if index 0 worked).")
        else:
            print("No USB cameras found. Please check connections and drivers.")

    def check_video(self):
        self.camera_capture = cv2.VideoCapture(self.img_source)
        if not self.camera_capture.isOpened():
            print(f"Error: Could not open video file: {self.img_source}")
            sys.exit(0)
        if self.user_res:
            self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resW)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resH)

    def check_picamera(self):
        try:
            pygame.init()
            pygame.camera.init()
            cam_list = pygame.camera.list_cameras()
            if not cam_list:
                print("No camera found. Please check the camera connection or try closing the application.")
                pygame.quit()
                sys.exit(0)
            else:
                print("Available cameras: ")
                print("\n " + str(cam_list))
                camera_name = cam_list[0]
                if self.user_res:
                    try:
                        self.resW, self.resH = map(int, self.user_res.split('x'))
                        self.camera_capture = pygame.camera.Camera(camera_name, (self.resW, self.resH))
                    except ValueError:
                        print('Invalid resolution format for picamera. Using default.')
                        self.camera_capture = pygame.camera.Camera(camera_name) # Use default resolution
                else:
                    self.camera_capture = pygame.camera.Camera(camera_name)
                self.camera_capture.start() # Start the camera

        except Exception as e:
            print(f"Error initializing PiCamera: {e}")
            if 'pygame' in sys.modules:
                pygame.quit()
            sys.exit(0)

    def check_usb(self):
        self.camera_capture = cv2.VideoCapture(self.usb_idx)
        if not self.camera_capture.isOpened():
            print(f"Error: Could not open USB camera with index: {self.usb_idx}")
            sys.exit(0)
        if self.user_res:
            self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resW)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resH)

    def run_inference(self):
        while True:
            self.t_start = time.perf_counter()
            frame = None

            if self.source_type == 'image' or self.source_type == 'folder':
                if self.img_count >= len(self.imgs_list):
                    print('All images have been processed. Exiting program.')
                    break
                img_filename = self.imgs_list[self.img_count]
                frame = cv2.imread(img_filename)
                if frame is None:
                    print(f"Warning: Could not read image file: {img_filename}. Skipping.")
                    self.img_count += 1
                    continue
                self.img_count += 1

            elif self.source_type == 'video' or self.source_type == 'usb':
                if self.camera_capture is None or not self.camera_capture.isOpened():
                    print(f"Error: Camera/Video source not available or closed. Exiting.")
                    break
                ret, frame = self.camera_capture.read()
                if not ret or frame is None:
                    print('Reached end of the video file or unable to read frame. Exiting program.')
                    break

            elif self.source_type == 'picamera':
                if self.camera_capture is None:
                    print("Error: PiCamera not initialized. Exiting.")
                    break
                try:
                    # Capture image and convert to numpy array, then to BGR format for OpenCV
                    img_pil = self.camera_capture.get_image()
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Error capturing frame from PiCamera: {e}")
                    break

            if frame is None:
                print("Error: Failed to get a valid frame. Exiting.")
                break

            if self.resize and frame is not None:
                frame = cv2.resize(frame, (self.resW, self.resH))

            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                classidx = int(detections[i].cls.item())
                classname = self.labels.get(classidx, f'Class_{classidx}') # Use .get for safer label lookup

                conf = detections[i].conf.item()

                if conf >= self.min_thresh:
                    color = self.bbox_colors[classidx % len(self.bbox_colors)] # Use len for dynamic color list
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f'{classname}: {int(conf * 100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    object_count += 1

            if self.source_type in ['video', 'usb', 'picamera']:
                cv2.putText(frame, f'FPS: {self.avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

            cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow('YOLO detection results', frame)

            if self.record:
                if self.recorder is not None:
                    self.recorder.write(frame)
                else:
                    print("Error: Recorder not initialized, cannot write frame.")


            self.wait_time = 0
            if self.source_type == 'image' or self.source_type == 'folder':
                self.wait_time = 0 # Wait indefinitely for key press
            else:
                self.wait_time = 5 # Wait 5 ms for video/camera feeds

            key = cv2.waitKey(self.wait_time) & 0xFF
            if key == ord('q'):
                break

            self.calculate_fps() # Moved this here to be after the frame is processed

    def clean_up(self):
        print(f'Average pipeline FPS: {self.avg_frame_rate:.2f}')
        if self.camera_capture is not None:
            if self.source_type == 'video' or self.source_type == 'usb':
                self.camera_capture.release()
            elif self.source_type == 'picamera':
                self.camera_capture.stop()
        if self.recorder is not None:
            self.recorder.release()
        cv2.destroyAllWindows()
        if 'pygame' in sys.modules:
            pygame.quit()

    def check_model_path(self):
        if self.check_file_path(self.model_path):
            print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
            sys.exit(0)

    def main(self):
        self.init_parse()
        self.check_model_path()

        try:
            self.model = YOLO(self.model_path, task='detect')
            self.labels = self.model.names
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(0)

        self.check_img_src()

        if self.user_res:
            self.resize = True
            try:
                self.resW, self.resH = map(int, self.user_res.split('x'))
            except ValueError:
                print('Invalid resolution format. Please use WxH (e.g., "640x480").')
                sys.exit(0)

        if self.record:
            if self.source_type not in ['video', 'usb', 'picamera']:
                print('Recording only works for video, USB camera, and Picamera sources. Please try again.')
                sys.exit(0)
            if not self.user_res:
                print('Please specify resolution to record video at using --resolution argument.')
                sys.exit(0)

            record_name = 'demo1.mp4'
            record_fps = 30 # You might want to make this dynamic or configurable

            if self.resW is None or self.resH is None:
                print("Error: Recording requires a resolution to be specified with --resolution.")
                sys.exit(0)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for broader compatibility
            self.recorder = cv2.VideoWriter(record_name, fourcc, record_fps, (self.resW, self.resH))
            if not self.recorder.isOpened():
                print(f"Error: Could not open VideoWriter for recording. Check codec and path.")
                sys.exit(0)
            print(f"Recording enabled. Output file: {record_name}")

        if self.source_type == 'image':
            self.imgs_list = [self.img_source]
        elif self.source_type == 'folder':
            self.check_folder()
        elif self.source_type == 'video':
            self.check_video()
        elif self.source_type == 'usb':
            self.check_usb()
        elif self.source_type == 'picamera':
            self.check_picamera()

        # Set bounding box colors (Tableau 10 color scheme)
        self.bbox_colors = [
            (230, 159, 2), (86, 180, 233), (246, 158, 107), (206, 114, 100), (178, 178, 178),
            (102, 194, 164), (252, 208, 162), (106, 61, 154), (148, 206, 114), (190, 87, 40)
        ]

        self.run_inference()
        self.clean_up()


f = Func()
f.main()
