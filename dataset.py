import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose

# CSV file to store data
csv_file = open('posture_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write header for all 33 landmarks (x, y, z)
header = []
for i in range(33):
    header += [f'x_{i}', f'y_{i}', f'z_{i}']
header.append('posture')
csv_writer.writerow(header)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            # Extract pose landmarks
            landmarks = result.pose_landmarks.landmark
            
            # Display landmarks on the image
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get user input for labeling
            cv2.putText(image, 'Press "s" for Standing, "i" for Sitting, "l" for Lying Down', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Posture Detection', image)

            key = cv2.waitKey(1)
            label = None
            if key == ord('s'):
                label = 'Standing'
            elif key == ord('i'):
                label = 'Sitting'
            elif key == ord('l'):
                label = 'Lying Down'

            # If a label is chosen, save the landmarks and label to CSV
            if label:
                row = []
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z]
                row.append(label)
                csv_writer.writerow(row)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
csv_file.close()
cap.release()
cv2.destroyAllWindows()
