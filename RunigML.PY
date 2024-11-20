import cv2
import mediapipe as mp
import numpy as np
import joblib  # For loading the saved model

# Load the trained model
model = joblib.load("random_forest_model.joblib")  # Replace with the actual file name of your saved model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

# Updated recognize_posture function to use the trained model
def recognize_posture(pose_landmarks):
    # Extract pose landmarks into a flat list
    landmarks = []
    for landmark in pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])  # Use (x, y, z)

    # Convert to NumPy array and reshape for prediction
    landmarks_array = np.array(landmarks).reshape(1, -1)  # Shape (1, num_features)

    # Predict posture using the trained model
    predicted_label = model.predict(landmarks_array)[0]

    # Map numeric labels to human-readable strings
    label_mapping = {
        0: "Sitting",
        1: "Standing"
    }

    # Return the corresponding posture as a string
    return label_mapping.get(predicted_label, "Unknown")





# Capture from webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Preprocess the image
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        holistic_results = holistic.process(image_rgb)
        face_results = face_detection.process(image_rgb)

        # Draw landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

        # Hand gesture recognition
        

        # Posture recognition
        if pose_results.pose_landmarks:
             posture = recognize_posture(pose_results.pose_landmarks)
             cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
 


        # Face detection
        
        # Display the image
        cv2.imshow('Gesture, Posture, and Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
