import cv2
import mediapipe as mp
import time
import numpy as np

class model():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
    
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
    

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920//4)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080//4)
dataset = []

h = model()
prev = time.time()
history = []


while True:
    # Read frame, make it rgb and flip
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    # Find keypoints, draw them and get back useful data
    h.findHands(image)
    image = h.drawHands(image)
    data = h.getKeypoints()

    pol = h.pos_thumb
    ind = h.pos_idx
    mean = (pol[0]+ind[0])/2, (pol[1]+ind[1])/2 
    dist = ((pol[0]-ind[0])**2 + (pol[1]-ind[1])**2)**0.5

    if dist < 50:
        cv2.circle(image, (int(mean[0]), int(mean[1])), 10, (255, 255, 255), -1)
    elif dist > 50 and dist < 400:
        cv2.circle(image, (int(mean[0]), int(mean[1])), 10, (0, 0, 0), -1)
    
    
    # Reshape bgr image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (1280, 720))
    
    # find centers
    if len(data["centers"]) > 0:
        for x, y, z in data["centers"]:
            image = cv2.circle(image, (int(x*1280), int(y*720)), 10, (255, 255, 255), -1)


    tuttiPugni = True
    n = 0
    
    
    
    cv2.putText(image, str(round(1/(time.time()-prev), 1)), (20, 50+50*(n+2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    prev = time.time()
    
    cv2.imshow('Hand Tracking', cv2.resize(image, (1600//2, 900//2)))

    # Step 10: Break the loop if the user presses the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    
    
    


# Step 9: Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

