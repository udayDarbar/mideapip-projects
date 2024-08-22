import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

# Define gesture recognition function
def recognize_gesture(landmarks):
    if (landmarks[mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y and
        landmarks[mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[mp_holistic.HandLandmark.WRIST].y):
        return "Thumbs Up"
    elif landmarks[mp_holistic.HandLandmark.WRIST].visibility < 0.9:
        return "Hand Waving"
    elif (landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y and
          landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.PINKY_TIP].y):
        return "Peace Sign"
    elif landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y:
        return "Pointing"
    else:
        visible_fingers = sum(1 for lm in landmarks if lm.visibility > 0.9)
        return f"{visible_fingers} Finger(s)"

# Define posture recognition function
def recognize_posture(pose_landmarks):
    hip_landmarks = [pose_landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]]
    if all(hip.y > 0.7 for hip in hip_landmarks):
        return "Sitting"
    else:
        return "Standing"

# File paths
input_video_path = 'VID_20240822_161431.mp4'  # Replace with your input video file path
output_video_path = 'output_video.mp4'  # Replace with your desired output video file path

# Open video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Process image and find landmarks
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        holistic_results = holistic.process(image_rgb)
        face_results = face_detection.process(image_rgb)

        # Draw pose and hand landmarks on the image
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

        # Recognize gesture
        if holistic_results.right_hand_landmarks:
            gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
            cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Recognize posture
        if pose_results.pose_landmarks:
            posture = recognize_posture(pose_results.pose_landmarks)
            cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Count faces detected
        if face_results.detections:
            num_faces = len(face_results.detections)
            cv2.putText(image, f"Detected {num_faces} Face(s)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(image)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
