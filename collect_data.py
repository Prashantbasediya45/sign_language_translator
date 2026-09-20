# collect_data.py - FIXED VERSION
import cv2
import mediapipe as mp
import csv
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

signs = ['Hello', 'I Love You', 'No', 'Please', 'Thank You', 'Yes']
data = []

# Check camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot access camera!")
    exit()

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

current_sign = None
sample_count = 0
max_samples = 100

print("="*60)
print("SIGN LANGUAGE DATA COLLECTION")
print("="*60)
print("\nPress keys 0-5 to start recording each sign:")
for i, sign in enumerate(signs):
    print(f"  [{i}] {sign}")
print("\nPress 'q' to quit and save\n")
print("="*60)

recording_started = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read from camera")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Display instructions and status
    if current_sign is not None:
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 255, 0), -1)
        cv2.putText(frame, f"Recording: {signs[current_sign]}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(frame, f"Samples: {sample_count}/{max_samples}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        recording_started = True
    else:
        cv2.rectangle(frame, (0, 0), (w, 60), (200, 200, 200), -1)
        cv2.putText(frame, "Press 0-5 to start | Q to quit", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Draw hands
    hands_detected = False
    if results.multi_hand_landmarks:
        hands_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            
            # If recording, save landmarks
            if current_sign is not None and sample_count < max_samples:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Save as: [label, x0, x1, ..., x20, y0, y1, ..., y20, z0, z1, ..., z20]
                data.append([current_sign] + landmarks)
                sample_count += 1
                
                if sample_count >= max_samples:
                    print(f"✓ Completed recording '{signs[current_sign]}' - {max_samples} samples")
                    current_sign = None
                    sample_count = 0
    
    # Warning if no hands detected while recording
    if current_sign is not None and not hands_detected:
        cv2.putText(frame, "NO HANDS DETECTED!", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('Sign Language Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key in [ord(str(i)) for i in range(6)]:
        sign_idx = int(chr(key))
        current_sign = sign_idx
        sample_count = 0
        print(f"\n>>> Recording: {signs[current_sign]} (hold sign steady)...")

cap.release()
cv2.destroyAllWindows()
hands.close()

# Save data with proper error checking
print("\n" + "="*60)
if len(data) == 0:
    print("WARNING: No data was collected!")
    print("Make sure:")
    print("  1. Your hands were visible to the camera")
    print("  2. You pressed keys 0-5 to start recording")
    print("  3. Lighting was adequate")
else:
    print(f"Collected {len(data)} total samples")
    
    # Create CSV with proper header
    try:
        with open('sign_data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header: label, x0-x20, y0-y20, z0-z20
            header = ['label'] + \
                     [f'x{i}' for i in range(21)] + \
                     [f'y{i}' for i in range(21)] + \
                     [f'z{i}' for i in range(21)]
            
            writer.writerow(header)
            writer.writerows(data)
        
        print(f"✓ Data saved to 'sign_data.csv'")
        print(f"  File size: {os.path.getsize('sign_data.csv')} bytes")
        
        # Show breakdown
        from collections import Counter
        label_counts = Counter([row[0] for row in data])
        print("\nSamples per sign:")
        for sign_idx, count in sorted(label_counts.items()):
            print(f"  {signs[sign_idx]}: {count} samples")
            
    except Exception as e:
        print(f"ERROR saving file: {e}")

print("="*60)