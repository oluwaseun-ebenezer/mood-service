import ultralytics
import numpy as np
from roboflow import Roboflow
from typing import Tuple, Union
import math
import cv2
import os
import numpy as np

# media pipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


dir = os.getcwd()

print(dir)

ultralytics.checks()

# load models
attention_model = ultralytics.YOLO(dir + '/models/best.pt')

# load model from roboflow
rf = Roboflow(api_key="jUksJerIsafXNaR1DLEz")
project = rf.workspace().project("faceemo")
model = project.version(2).model


# Create an FaceDetector object.
base_options = python.BaseOptions(
    model_asset_path=f"{dir}/models/detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

# HELPER FUNCTIONS BLOCK


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def apply_attention_detection_model(img):
    global attention_model
    predictions = attention_model.predict(img)
    attentive_count = 0
    drowsy_count = 0
    for prediction in predictions:
        labels = prediction.names
        classes_detected = np.array(prediction.boxes.cls.cpu(), dtype=np.int32)
        print(prediction.boxes.data, 'attention detected',
              classes_detected, labels)
        for cls in classes_detected:
            if (cls == 0):
                attentive_count += 1
            else:
                drowsy_count += 1
        if (len(classes_detected) >= 1):
            return f"{(attentive_count / len(classes_detected)) * 100}% attentive: {(drowsy_count / len(classes_detected)) * 100}% drowsy"
        return ""


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    global attention_model
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
        box_region = annotated_image[start_point[1]                                     :end_point[1], start_point[0]:end_point[0]]
        cv2.imshow('detected', box_region)
        # pass box_region to your attention detection model
        result_atten = apply_attention_detection_model(box_region)
        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + \
            ' (' + str(probability) + ')' + result_atten
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


def apply_face_detection_inference(image):
    global detector
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # Detect faces in the input image.
    detection_result = detector.detect(mp_image)
    print(detection_result)
    print("=============")

    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('annotated', rgb_annotated_image)

#### video aspect###

# break video down into images


engaged_count = 0
boredom_count = 0
confusion_count = 0
frustration_count = 0
distracted_count = 0


video = cv2.VideoCapture(dir + '/istockphoto-1336667150-640_adpp_is.mp4')
# video = cv2.VideoCapture(1)
# read video frames
print(video)
while True:
    flag, frame = video.read()
    print(flag, frame)
    if flag:
        # The frame is ready and already captured
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        apply_face_detection_inference(frame_rgb)
        key = cv2.waitKey(100)
        if key == 27:
            break
    else:
        break


# sum_inference = engaged_count + boredom_count + \
#     confusion_count + frustration_count+distracted_count
# engaged_percent = (engaged_count/sum_inference)*100
# boredom_percent = (boredom_count/sum_inference)*100
# confusion_percent = (confusion_count/sum_inference)*100
# frustration_percent = (frustration_count/sum_inference)*100
# distracted_percent = (distracted_count/sum_inference)*100

# print(
#     f"students were engaged {engaged_count}times which was {engaged_percent} of {sum_inference}")
# print(
#     f"students were bored {boredom_count}times which was {boredom_percent} of {sum_inference}")
# print(
#     f"students were confused {confusion_count}times which was {confusion_percent} of {sum_inference}")
# print(
#     f"students were frustrated {frustration_count}times which was {frustration_percent} of {sum_inference}")
# print(
#     f"students were distracted {distracted_count}times which was {distracted_percent} of {sum_inference}")
