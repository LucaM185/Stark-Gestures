import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
import threading
from collections import deque  # Add this import for the position history

class model():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Update to track multiple hands (default is 2)
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,  # Changed from 1 to 2 to track multiple hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Keep this for performance
        )
        self.pos_thumb = None
        self.pos_idx = None
    
    def findHands(self, image):
        self.results = self.hands.process(image)
        
    def drawHands(self, image):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # check if landmark is tip of index
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y:
                    image = cv2.circle(image, (int(hand_landmarks.landmark[8].x*image.shape[1]), int(hand_landmarks.landmark[8].y*image.shape[0])), 10, (0, 255, 0), -1)
                    self.pos_idx = (int(hand_landmarks.landmark[8].x*image.shape[1]), int(hand_landmarks.landmark[8].y*image.shape[0]))
                else:
                    image = cv2.circle(image, (int(hand_landmarks.landmark[8].x*image.shape[1]), int(hand_landmarks.landmark[8].y*image.shape[0])), 10, (0, 0, 255), -1)
                    self.pos_idx = (int(hand_landmarks.landmark[8].x*image.shape[1]), int(hand_landmarks.landmark[8].y*image.shape[0]))
                # check if landmark is tip of thumb
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    image = cv2.circle(image, (int(hand_landmarks.landmark[4].x*image.shape[1]), int(hand_landmarks.landmark[4].y*image.shape[0])), 10, (0, 255, 0), -1)
                    self.pos_thumb = (int(hand_landmarks.landmark[4].x*image.shape[1]), int(hand_landmarks.landmark[4].y*image.shape[0]))
                else:
                    image = cv2.circle(image, (int(hand_landmarks.landmark[4].x*image.shape[1]), int(hand_landmarks.landmark[4].y*image.shape[0])), 10, (0, 0, 255), -1)
                    self.pos_thumb = (int(hand_landmarks.landmark[4].x*image.shape[1]), int(hand_landmarks.landmark[4].y*image.shape[0]))
                
        return image
    
    def getKeypoints(self):
        data = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    data.append([landmark.x, landmark.y, landmark.z])
        data = np.array(data)
        
        hands = []
        centers = []
        for i in range(0, data.shape[0], 21):
            hands.append(data[i:i+21])
            centers.append(np.mean(hands[-1], axis = 0).reshape(3))
        hands = np.array(hands).reshape(-1, 21, 3)
        centers = np.array(centers).reshape(-1, 3)
            
        if len(centers.shape) == 2 and centers.shape[0] >= 2:
            dist = (np.sum((centers[0, :2] - centers[1, :2])**2))**0.5
        else:
            dist = None
        
        dataset = {
            "hands": hands,
            "centers": centers,
            "distance": dist
        }
        return dataset

# Camera thread class to separate capture from processing
class CameraThread:
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.ret = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.running = False

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = ret

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# Use lower resolution for better performance
cam_thread = CameraThread(width=640, height=480).start()
time.sleep(1.0)  # Give camera time to initialize
dataset = []

h = model()
prev = time.time()
frame_count = 0
process_every_n_frames = 1  # Start with processing all frames

# Before the main loop, get screen dimensions
screen_width, screen_height = pyautogui.size()

# Add variables for cursor position smoothing
prev_x, prev_y = 0, 0
smoothing = 4  # Lower is more responsive, higher is smoother

# Add click state tracking variables
previous_finger_state = "open"
click_debounce_time = 0
debounce_threshold = 0.5

# Position history for delayed clicks (stores tuples of (timestamp, x, y))
position_history = deque(maxlen=30)  # Store ~1 second of history assuming ~30fps
history_delay = 0.1  # Delay in seconds for click position

# Flags for toggling display components
show_debugging = True

while True:
    # Use threaded camera capture
    ret, frame = cam_thread.read()
    if not ret:
        continue
    
    frame_count += 1
    current_time = time.time()
    
    # Process only every n frames to increase speed if needed
    if frame_count % process_every_n_frames == 0:
        # Convert and flip once
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        # Find keypoints - this is the most expensive operation
        h.findHands(image)
        
        if h.results.multi_hand_landmarks:
            # Only draw if we have hands (save processing)
            image = h.drawHands(image)
            
            # Get keypoints only when we need them
            if h.pos_thumb is not None and h.pos_idx is not None:
                pol = h.pos_thumb
                ind = h.pos_idx
                mean = (pol[0]+ind[0])/2, (pol[1]+ind[1])/2 
                dist = ((pol[0]-ind[0])**2 + (pol[1]-ind[1])**2)**0.5

                # Cursor mapping
                margin_x = image.shape[1] * 0.2
                margin_y = image.shape[0] * 0.2
                
                cursor_x = np.interp(mean[0], 
                                [margin_x, image.shape[1] - margin_x], 
                                [0, screen_width])
                cursor_y = np.interp(mean[1], 
                                [margin_y, image.shape[0] - margin_y], 
                                [0, screen_height])
                
                # Apply smoothing
                prev_x = prev_x + (cursor_x - prev_x) / smoothing
                prev_y = prev_y + (cursor_y - prev_y) / smoothing
                
                # Store current position with timestamp in history
                position_history.append((time.time(), prev_x, prev_y))
                
                # Move cursor
                try:
                    pyautogui.moveTo(prev_x, prev_y, duration=0.01, _pause=False)
                except Exception as e:
                    print(e)
                    
                # Handle clicks with debouncing
                if dist < 50:
                    if previous_finger_state == "open" and (current_time - click_debounce_time > debounce_threshold):
                        # Find the historical position to click at
                        target_time = current_time - history_delay
                        click_x, click_y = prev_x, prev_y  # Default to current position
                        
                        # Improved search for position closest to target time
                        if position_history:
                            best_time_diff = float('inf')
                            
                            for hist_time, hist_x, hist_y in position_history:
                                time_diff = abs(hist_time - target_time)
                                if time_diff < best_time_diff:
                                    best_time_diff = time_diff
                                    click_x, click_y = hist_x, hist_y
                            
                            # Debug: visualize where the historical click will happen
                            cv2.circle(image, (int(click_x * image.shape[1] / screen_width), 
                                              int(click_y * image.shape[0] / screen_height)), 
                                      15, (255, 0, 0), 2)
                        
                        # Click at historical position
                        current_x, current_y = pyautogui.position()
                        pyautogui.click(x=int(click_x), y=int(click_y))
                        # Move back to current position after clicking
                        pyautogui.moveTo(current_x, current_y, _pause=False)
                        
                        click_debounce_time = current_time
                        cv2.putText(image, f"CLICK at {int(click_x)},{int(click_y)}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    previous_finger_state = "closed"
                elif dist > 50:
                    previous_finger_state = "open"

        # Only convert back to BGR once for display
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Only resize once for display
        display_image = cv2.resize(display_image, (800, 600))
        
        # Display FPS
        elapsed = time.time() - prev
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(display_image, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prev = time.time()
        
        # Display image (avoid multiple resizing)
        cv2.imshow('Hand Tracking', display_image)
    
    # Adjust processing rate dynamically based on performance
    if frame_count % 30 == 0:  # Check every 30 frames
        elapsed_time = time.time() - prev
        current_fps = 30 / elapsed_time if elapsed_time > 0 else 0
        
        # If FPS is too low, process fewer frames
        if current_fps < 10 and process_every_n_frames < 3:
            process_every_n_frames += 1
        # If FPS is good, process more frames
        elif current_fps > 20 and process_every_n_frames > 1:
            process_every_n_frames -= 1
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Toggle debug display with 'd' key
    if cv2.waitKey(1) & 0xFF == ord('d'):
        show_debugging = not show_debugging

# Release resources
cam_thread.release()
cv2.destroyAllWindows()

