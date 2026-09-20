# translator_app.py - FIXED VERSION
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
from collections import deque
import time
import json

# Initialize
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
@st.cache_resource
def load_model():
    model = joblib.load('sign_model.pkl')
    with open('sign_labels.json', 'r') as f:
        label_mapping = json.load(f)
    # Convert string keys back to integers
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    return model, label_mapping

@st.cache_resource
def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        return engine
    except:
        return None

# Streamlit UI
st.set_page_config(page_title="Sign Language Translator", layout="wide")
st.title("🤟 Real-Time Sign Language Translator")
st.markdown("Hold a sign for 2 seconds to add it to your sentence")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📝 Translated Sentence")
    sentence_display = st.empty()
    st.subheader("🎯 Current Prediction")
    prediction_display = st.empty()
    confidence_display = st.empty()
    
    if st.button("🔊 Speak Sentence", use_container_width=True):
        if 'sentence' in st.session_state and st.session_state.sentence:
            tts = init_tts()
            if tts:
                try:
                    tts.say(st.session_state.sentence)
                    tts.runAndWait()
                except:
                    st.error("Text-to-speech failed")
            else:
                st.warning("Text-to-speech not available")
    
    if st.button("🗑️ Clear Sentence", use_container_width=True):
        st.session_state.sentence = ""
    
    st.markdown("---")
    st.subheader("Available Signs")
    
    # Load and display available signs
    try:
        model, label_mapping = load_model()
        for label_idx in sorted(label_mapping.keys()):
            st.markdown(f"• {label_mapping[label_idx]}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Initialize session state
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# Main video processing
with col1:
    video_placeholder = st.empty()
    
    model, label_mapping = load_model()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access camera. Please check your camera connection.")
        st.stop()
    
    # For sign stability detection
    prediction_buffer = deque(maxlen=60)  # 2 seconds at 30fps
    last_added_sign = None
    last_add_time = 0
    
    run = st.checkbox("Start Camera", value=True)
    stop_button = st.button("Stop")
    
    frame_count = 0
    
    while run and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_prediction = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                # Extract features
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Predict
                features = np.array(landmarks).reshape(1, -1)
                pred_label = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                
                # Get prediction index in probability array
                pred_idx = list(model.classes_).index(pred_label)
                
                current_prediction = label_mapping[pred_label]
                confidence = proba[pred_idx]
                
                # Add to buffer
                prediction_buffer.append((current_prediction, confidence))
                
                # Display prediction on frame
                cv2.putText(frame, f"{current_prediction} ({confidence*100:.0f}%)",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5, (0, 255, 0), 3)
        else:
            # No hands detected
            cv2.putText(frame, "No hands detected",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        # Check if sign is stable for 2 seconds
        if len(prediction_buffer) == 60:
            predictions = [p[0] for p in prediction_buffer]
            confidences = [p[1] for p in prediction_buffer]
            
            # If same prediction for 90% of buffer with high confidence
            most_common = max(set(predictions), key=predictions.count)
            if predictions.count(most_common) > 54 and np.mean(confidences) > 0.7:
                stable_sign = most_common
                current_time = time.time()
                
                # Add to sentence if different from last or enough time passed
                if (stable_sign != last_added_sign or 
                    current_time - last_add_time > 3):
                    st.session_state.sentence += f" {stable_sign}"
                    last_added_sign = stable_sign
                    last_add_time = current_time
                    prediction_buffer.clear()
        
        # Update displays
        sentence_display.markdown(f"### {st.session_state.sentence}")
        if current_prediction:
            prediction_display.markdown(f"**{current_prediction}**")
            confidence_display.progress(float(confidence))
        else:
            prediction_display.markdown("*Waiting for hand...*")
            confidence_display.progress(0.0)
        
        # Show video (every 2nd frame for performance)
        frame_count += 1
        if frame_count % 2 == 0:
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")
st.markdown("💡 **Tips:**")
st.markdown("• Keep hands clearly visible with good lighting")
st.markdown("• Hold each sign steady for 2 seconds to add to sentence")
st.markdown("• Different signs need 3+ seconds gap between them")