import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
from collections import deque
import pyautogui  # For media control (play/pause, skip)

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize Windows Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# FPS Counter Setup
prev_time = 0
fps = 0

# Smoothing Setup
distance_buffer = deque(maxlen=5)

# ========== GESTURE DETECTION VARIABLES ==========
# Track previous hand position for swipe detection
prev_hand_x = None
swipe_threshold = 100  # Pixels to move for swipe detection
last_swipe_time = 0  # Prevent multiple swipes
swipe_cooldown = 1.0  # 1 second cooldown between swipes

# Track gesture state to prevent repeated actions
last_gesture = None
gesture_cooldown = 0.5  # 0.5 seconds between gesture actions
last_gesture_time = 0

# Track if currently muted
is_muted = False
volume_before_mute = 0

print("=== Advanced Gesture Volume Control ===")
print("Gestures:")
print(" Pinch (Thumb + Index) = Volume Control")
print(" Closed Fist = Mute/Unmute")
print("  Peace Sign (2 fingers) = Play/Pause Media")
print("Thumbs Up = Volume to 50%")
print(" Open Palm (5 fingers) = Volume to Max")
print(" Swipe Left = Previous Track")
print(" Swipe Right = Next Track")
print("\nPress 'q' to quit")
print("=======================================\n")


def draw_volume_bar(img, vol_percentage):
    """Draws a vertical volume bar on the right side"""
    bar_x = 580
    bar_y = 100
    bar_width = 40
    bar_height = 300

    # Draw background
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (50, 50, 50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 3)

    filled_height = int((vol_percentage / 100) * bar_height)

    if vol_percentage < 33:
        color = (0, 0, 255)
    elif vol_percentage < 66:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)

    if filled_height > 0:
        cv2.rectangle(img,
                      (bar_x, bar_y + bar_height - filled_height),
                      (bar_x + bar_width, bar_y + bar_height),
                      color, -1)

    cv2.putText(img, f'{int(vol_percentage)}%',
                (bar_x - 10, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


def count_fingers(landmarks, hand_label):
    """
    Count extended fingers based on landmark positions
    Returns: number of extended fingers (0-5)
    """
    fingers = []

    # Thumb detection (different logic because thumb moves differently)
    # Compare thumb tip (4) with thumb IP joint (3)
    if hand_label == "Right":
        # For right hand, thumb extends to the right
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # For left hand, thumb extends to the left
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers: compare tip with PIP joint (2 joints down)
    # Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        # If tip is above PIP joint, finger is extended
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def detect_gesture(landmarks, hand_label):
    """
    Detect specific hand gestures
    Returns: gesture name as string
    """
    finger_count = count_fingers(landmarks, hand_label)

    # Calculate distances for pinch and thumbs up detection
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    # Distance between thumb and index
    thumb_index_dist = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )

    # Distance between thumb and middle finger
    thumb_middle_dist = math.sqrt(
        (thumb_tip.x - middle_tip.x) ** 2 +
        (thumb_tip.y - middle_tip.y) ** 2
    )

    # GESTURE DETECTION LOGIC

    # Closed Fist: 0 fingers extended
    if finger_count == 0:
        return "fist"

    # Peace Sign: 2 fingers (index and middle)
    # Check if only index and middle are up
    if finger_count == 2:
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        if index_up and middle_up:
            return "peace"

    # Thumbs Up: Only thumb extended, pointing up
    if finger_count == 1:
        thumb_up = landmarks[4].y < landmarks[3].y
        if thumb_up:
            return "thumbs_up"

    # Open Palm: All 5 fingers extended
    if finger_count == 5:
        return "open_palm"

    # Pinch: Thumb and index close together
    if thumb_index_dist < 0.05:  # Normalized distance
        return "pinch"

    return "unknown"


def execute_gesture_action(gesture, current_time):
    """
    Execute action based on detected gesture
    Uses cooldown to prevent rapid repeated actions
    """
    global last_gesture, last_gesture_time, is_muted, volume_before_mute

    # Check cooldown
    if current_time - last_gesture_time < gesture_cooldown:
        return False

    # Only execute if gesture changed (except for pinch which is continuous)
    if gesture == last_gesture and gesture != "pinch":
        return False

    action_executed = False

    if gesture == "fist":
        # Mute/Unmute toggle
        if is_muted:
            # Unmute: restore previous volume
            volume.SetMasterVolumeLevel(volume_before_mute, None)
            is_muted = False
            print(" UNMUTED")
        else:
            # Mute: save current volume and set to minimum
            volume_before_mute = volume.GetMasterVolumeLevel()
            volume.SetMasterVolumeLevel(min_vol, None)
            is_muted = True
            print(" MUTED")
        action_executed = True

    elif gesture == "peace":
        # Play/Pause media
        pyautogui.press('playpause')
        print("⏯  PLAY/PAUSE")
        action_executed = True

    elif gesture == "thumbs_up":
        # Set volume to 50%
        mid_vol = (min_vol + max_vol) / 2
        volume.SetMasterVolumeLevel(mid_vol, None)
        print(" VOLUME → 50%")
        action_executed = True

    elif gesture == "open_palm":
        # Set volume to maximum
        volume.SetMasterVolumeLevel(max_vol, None)
        print("VOLUME → MAX")
        action_executed = True

    if action_executed:
        last_gesture = gesture
        last_gesture_time = current_time
        return True

    return False


while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video")
        break

    # FPS Calculation
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    if time_diff > 0:
        fps = 1 / time_diff

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    vol_percentage = 0
    current_gesture = "none"

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark
            hand_label = hand_handedness.classification[0].label

            # ========== SWIPE DETECTION ==========
            # Get palm center (landmark 0)
            palm_x = int(landmarks[0].x * img.shape[1])

            if prev_hand_x is not None:
                x_diff = palm_x - prev_hand_x

                # Check if enough time passed since last swipe
                if current_time - last_swipe_time > swipe_cooldown:
                    # Swipe Right
                    if x_diff > swipe_threshold:
                        pyautogui.press('nexttrack')
                        print(" NEXT TRACK")
                        last_swipe_time = current_time
                    # Swipe Left
                    elif x_diff < -swipe_threshold:
                        pyautogui.press('prevtrack')
                        print(" PREVIOUS TRACK")
                        last_swipe_time = current_time

            prev_hand_x = palm_x

            # ========== GESTURE DETECTION ==========
            current_gesture = detect_gesture(landmarks, hand_label)

            # Execute gesture action
            execute_gesture_action(current_gesture, current_time)

            # ========== PINCH VOLUME CONTROL ==========
            if current_gesture == "pinch" or current_gesture == "unknown":
                # Get thumb and index positions
                thumb_x = int(landmarks[4].x * img.shape[1])
                thumb_y = int(landmarks[4].y * img.shape[0])
                index_x = int(landmarks[8].x * img.shape[1])
                index_y = int(landmarks[8].y * img.shape[0])

                # Draw circles and line
                cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

                # Calculate distance
                distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                # Smoothing
                distance_buffer.append(distance)
                smoothed_distance = sum(distance_buffer) / len(distance_buffer)

                # Midpoint visualization
                mid_x = (thumb_x + index_x) // 2
                mid_y = (thumb_y + index_y) // 2
                color = (0, 0, 255) if smoothed_distance < 50 else (0, 255, 0)
                cv2.circle(img, (mid_x, mid_y), 10, color, cv2.FILLED)

                # Set volume

                vol = np.interp(smoothed_distance, [20, 100], [min_vol, max_vol])
                vol_percentage = np.interp(smoothed_distance, [20, 200], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)
                print("Distance:", smoothed_distance, "Volume %:", vol_percentage)

    # ========== DISPLAY INFORMATION ==========
    # FPS
    cv2.putText(img, f'FPS: {int(fps)}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)

    # Current Gesture
    gesture_display = current_gesture.replace("_", " ").title()
    cv2.putText(img, f'Gesture: {gesture_display}',
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    # Mute Status
    if is_muted:
        cv2.putText(img, 'MUTED',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # Volume Bar
    draw_volume_bar(img, vol_percentage)

    cv2.imshow('Advanced Gesture Control', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

