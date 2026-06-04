import threading
import time
import numpy as np
import cv2

# We import Picamera2 but wrap it in a try-except so it doesn't crash if imported on a non-Pi machine
try:
    from picamera2 import Picamera2
except ImportError:
    print("Picamera2 not found. Make sure you are running this on a Raspberry Pi with libcamera installed.")
    Picamera2 = None

class DualCamera:
    def __init__(self, resolution=(1640, 1232), framerate=83, format="YUV420"):
        """
        Initializes a dual camera setup specifically tuned for the IMX219 on Raspberry Pi.
        We request YUV420 format so we can do high-speed color thresholding purely on the U/V channels.
        """
        self.resolution = resolution
        self.framerate = framerate
        self.format = format
        
        self.cam0 = None
        self.cam1 = None
        
        self.frame0 = None
        self.frame1 = None
        
        self.lock0 = threading.Lock()
        self.lock1 = threading.Lock()
        
        self.running = False
        
        if Picamera2 is not None:
            self._init_picamera2()
        else:
            self._init_opencv_fallback()

    def _init_picamera2(self):
        # Initialize each camera independently so one failure doesn't kill both
        for cam_idx in [0, 1]:
            try:
                cam = Picamera2(cam_idx)
                config = cam.create_preview_configuration(
                    main={"size": self.resolution, "format": self.format}
                )
                cam.configure(config)
                if cam_idx == 0:
                    self.cam0 = cam
                else:
                    self.cam1 = cam
                print(f"  Camera {cam_idx} initialized at {self.resolution}")
            except Exception as e:
                print(f"  WARNING: Camera {cam_idx} failed to initialize: {e}")
        
        if self.cam0 is None and self.cam1 is None:
            print("CRITICAL: No cameras initialized! Falling back to OpenCV.")
            self._init_opencv_fallback()

    def _init_opencv_fallback(self):
        print("Using standard OpenCV VideoCapture fallback...")
        self.cam0 = cv2.VideoCapture(0)
        self.cam0.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cam0.set(cv2.CAP_PROP_FPS, self.framerate)
        
        self.cam1 = cv2.VideoCapture(1)
        self.cam1.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cam1.set(cv2.CAP_PROP_FPS, self.framerate)

    def _capture_loop_0(self):
        while self.running:
            if Picamera2 is not None and hasattr(self.cam0, 'capture_array'):
                frame = self.cam0.capture_array()
            elif self.cam0 and self.cam0.isOpened():
                ret, frame = self.cam0.read()
                if not ret: continue
            else:
                continue
                
            with self.lock0:
                self.frame0 = frame

    def _capture_loop_1(self):
        while self.running:
            if Picamera2 is not None and hasattr(self.cam1, 'capture_array'):
                frame = self.cam1.capture_array()
            elif self.cam1 and self.cam1.isOpened():
                ret, frame = self.cam1.read()
                if not ret: continue
            else:
                continue
                
            with self.lock1:
                self.frame1 = frame

    def start(self):
        if Picamera2 is not None:
            if self.cam0: self.cam0.start()
            if self.cam1: self.cam1.start()
            
        self.running = True
        self.thread0 = threading.Thread(target=self._capture_loop_0, daemon=True)
        self.thread1 = threading.Thread(target=self._capture_loop_1, daemon=True)
        
        self.thread0.start()
        self.thread1.start()
        print("Dual camera capture started on background threads.")

    def stop(self):
        self.running = False
        if hasattr(self, 'thread0'):
            self.thread0.join()
        if hasattr(self, 'thread1'):
            self.thread1.join()
            
        if self.cam0:
            if Picamera2 is not None: self.cam0.stop()
            else: self.cam0.release()
        if self.cam1:
            if Picamera2 is not None: self.cam1.stop()
            else: self.cam1.release()
        print("Cameras stopped.")

    def _crop_stride(self, frame):
        """
        Reconstruct a clean I420 frame without stride padding.
        
        Picamera2 returns YUV420 (I420) frames where each row is padded to
        a stride boundary. In the I420 layout (shape: H*3/2, stride):
          - Y plane: H rows × stride (valid: W columns)
          - U plane: H/4 rows × stride; packs H/2 logical rows of stride/2,
            valid W/2 columns per logical row
          - V plane: same as U
        
        Simply cropping columns corrupts U/V (barcode artifacts) because
        two logical chroma rows share one physical row. We must unpack
        each plane, crop to valid width, and repack.
        """
        if frame is None:
            return None
        req_w = self.resolution[0]
        req_h = self.resolution[1]
        stride = frame.shape[1]
        
        if stride <= req_w:
            return frame  # No padding
        
        half_w = req_w // 2
        half_h = req_h // 2
        quarter_h = req_h // 4
        half_stride = stride // 2
        
        # Y plane: crop columns to requested width
        y = frame[:req_h, :req_w]
        
        # U plane: unpack 2 logical rows per physical row, crop, repack
        u_packed = frame[req_h : req_h + quarter_h, :]
        u = np.ascontiguousarray(
            u_packed.reshape(half_h, half_stride)[:, :half_w]
        )
        u_clean = u.reshape(quarter_h, req_w)
        
        # V plane: same treatment
        v_packed = frame[req_h + quarter_h : req_h + 2 * quarter_h, :]
        v = np.ascontiguousarray(
            v_packed.reshape(half_h, half_stride)[:, :half_w]
        )
        v_clean = v.reshape(quarter_h, req_w)
        
        return np.ascontiguousarray(np.vstack([y, u_clean, v_clean]))

    def get_frames(self):
        """Returns the latest (frame0, frame1) with stride padding removed."""
        f0 = None
        f1 = None
        with self.lock0:
            if self.frame0 is not None:
                f0 = self._crop_stride(self.frame0.copy())
        with self.lock1:
            if self.frame1 is not None:
                f1 = self._crop_stride(self.frame1.copy())
        return f0, f1

if __name__ == "__main__":
    cameras = DualCamera()
    cameras.start()
    
    # Wait for frames to populate
    time.sleep(1.0)
    
    start_time = time.time()
    frames_grabbed = 0
    while time.time() - start_time < 5.0:
        f0, f1 = cameras.get_frames()
        if f0 is not None and f1 is not None:
            frames_grabbed += 1
            time.sleep(0.01) # Simulating processing 
            
    print(f"Grabbed {frames_grabbed} frame pairs in 5 seconds.")
    cameras.stop()
