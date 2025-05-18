import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only need one hand for controls
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Variables for gesture detection
        self.prev_landmarks = None
        self.gesture_start_time = None
        self.swipe_threshold = 0.1  # Minimum movement to register a swipe
        self.jump_cooldown = 0.5  # Prevent spamming jump
        self.last_jump_time = 0

    def detect_swipe(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return None

        # Get wrist (landmark 0) and index tip (landmark 8) positions
        wrist_prev = np.array(self.prev_landmarks[0][:2])  # [x, y]
        wrist_curr = np.array(landmarks[0][:2])
        index_prev = np.array(self.prev_landmarks[8][:2])
        index_curr = np.array(landmarks[8][:2])

        # Calculate movement direction
        wrist_movement = wrist_curr - wrist_prev
        index_movement = index_curr - index_prev

        # Check if movement is significant
        if np.linalg.norm(wrist_movement) > self.swipe_threshold:
            direction = None
            if abs(wrist_movement[0]) > abs(wrist_movement[1]):  # Horizontal swipe
                if wrist_movement[0] > 0:
                    direction = "right"
                else:
                    direction = "left"
            else:  # Vertical swipe
                if wrist_movement[1] > 0:
                    direction = "down"
                else:
                    direction = "up"

            self.prev_landmarks = landmarks
            return direction
        return None

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Detect swipe direction
                direction = self.detect_swipe(landmarks)
                if direction:
                    current_time = time.time()
                    if direction == "up" and current_time - self.last_jump_time > self.jump_cooldown:
                        pyautogui.press('up')  # Jump
                        self.last_jump_time = current_time
                    elif direction == "down":
                        pyautogui.press('down')  # Roll
                    elif direction == "left":
                        pyautogui.press('left')  # Move left
                    elif direction == "right":
                        pyautogui.press('right')  # Move right

        return image

def main():
    controller = HandGestureController()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = controller.process_frame(frame)
        cv2.imshow('Subway Surfers Hand Control', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()