import warnings
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from dotenv import dotenv_values

from fer_methods import Lhc, Rmn, ResNet18
from filtering import get_most_frequent_value_in_queue, push
from utils import download_model, model_downloaded

config = dotenv_values(".env")
predictions = np.array([], dtype=np.int8)
most_frequent_class = None
prediction_time_sum = 0
predictions_counter = 0
prev_frame_time = 0
current_frame_time = 0
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Don't print warnings.
warnings.filterwarnings("ignore")

# Get desired camera as input, set font.
cap = cv2.VideoCapture(int(config["CAMERA"]))
font = cv2.FONT_HERSHEY_SIMPLEX

# Recognition model selection
print("Please input desired face recognition model.")
desired_model = input()
if desired_model == 0:
    model = Rmn()
else:
    model = Lhc()

# Download and load recognition model.
if not model_downloaded(model.filename):
    download_model(id=model.download_id, filename=model.filename)
model.load_model()

# Load detection model from Media Pipe.
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=int(config["DETECTION_MODEL"]),
    min_detection_confidence=float(config["MIN_DETECTION_CONFIDENCE"]),
) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Mark the image as not writeable to improve performance,
        # then it passes the image by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces on captured frame.
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, num_of_channels = image.shape

        # If face is detected, predict class for face on captured frame.
        if results.detections:
            try:
                for detection in results.detections:
                    face_bounding_box = detection.location_data.relative_bounding_box

                    # Get x, y coordinates from top left corner of bounding box.
                    x, y = int(face_bounding_box.xmin * image_width), int(
                        face_bounding_box.ymin * image_height
                    )

                    # Get height and width of bounding box.
                    bBox_width = x + int(face_bounding_box.width * image_width)
                    bBox_height = y + int(face_bounding_box.height * image_height)

                    # Draw bounding box to the original image.
                    cv2.rectangle(
                        img=image,
                        pt1=(x, y),
                        pt2=(bBox_width, bBox_height),
                        color=(255, 255, 255),
                        thickness=2,
                    )

                    # Cropp bounding box.
                    face_image = image[y:bBox_height, x:bBox_width]

                    # Adjust face image to suit model.
                    face_image = model.prepare_image_for_prediction(image=face_image)

                    # Predict emotion and measure prediction time.
                    start = perf_counter()
                    (
                        current_prediction,
                        classification_confidence,
                    ) = model.predict_emotion(image=face_image)
                    end = perf_counter()

                    # Don't measure first prediction time, it takes longer to predict.
                    if predictions_counter != 0:
                        prediction_time = end - start
                        prediction_time_sum += prediction_time

                        # Print average classification time.
                        cv2.putText(
                            img=image,
                            text=f"Avg class. time: {round(prediction_time_sum / predictions_counter, 3)} s",
                            org=(10, 50),
                            fontFace=font,
                            fontScale=0.5,
                            color=(100, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )
                    predictions_counter += 1

                    # Show prediction results if confidence is enough.
                    if classification_confidence > 65:
                        # Filtering.
                        predictions = push(
                            queue=predictions,
                            new_value=current_prediction,
                        )
                        most_frequent_class = get_most_frequent_value_in_queue(
                            queue=predictions
                        )

                    if most_frequent_class != None:
                        class_idx = most_frequent_class
                    else:
                        class_idx = current_prediction

                    # Print class.
                    cv2.putText(
                        img=image,
                        text=str(classes[class_idx]),
                        org=(x + 5, y - 5),
                        fontFace=font,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

            except:
                pass

        # Calculate FPS.
        current_frame_time = perf_counter()
        fps = 1 / (current_frame_time - prev_frame_time)
        prev_frame_time = current_frame_time
        fps = int(fps)

        # Print FPS.
        cv2.putText(
            img=image,
            text=f"FPS: {fps}",
            org=(10, 20),
            fontFace=font,
            fontScale=0.5,
            color=(100, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("Face emotion recognition", image)

        if cv2.waitKey(5) & 0xFF == 27:  # press 'ESC' to quit
            break

cap.release()
cv2.destroyAllWindows()
