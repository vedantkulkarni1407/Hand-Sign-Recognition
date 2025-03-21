import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('hand_sign_model.h5')

# Load label classes
label_classes = np.load('label_encoder_classes.npy', allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and normalize landmarks
            landmark_array = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
            
            # Center landmarks
            x_mean, y_mean = np.mean(landmark_array, axis=0)
            centered_landmarks = landmark_array - [x_mean, y_mean]
            
            # Scale landmarks
            max_dist = np.max(np.sqrt(np.sum(centered_landmarks**2, axis=1)))
            if max_dist > 0:
                scaled_landmarks = centered_landmarks / max_dist
            else:
                scaled_landmarks = centered_landmarks
            
            landmark_input = scaled_landmarks.flatten().reshape(1, 42)
            
            # Predict hand sign
            prediction = model.predict(landmark_input)
            predicted_label = label_classes[np.argmax(prediction)]
            confidence = prediction[0][np.argmax(prediction)]
            
            # Display prediction on frame
            cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Hand Sign Recognition', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()