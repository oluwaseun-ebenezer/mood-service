"""
    *************************************************************
    *                                                           *
    *   Project: Student-Engagement Detection                   *
    *                                                           *
    *   Program name: app.py                                    *
    *                                                           *
    *************************************************************

"""

"""
    ************************************
    *                                  *
    *     EMOTION DETECTION SECTION    *
    *                                  *
    ************************************
"""

# import module

# =============== TWO MODEL APPROACH ====================
# using face detection and attention detection

import time
from threading import Thread
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
import traceback
import ultralytics
import numpy as np
from time import sleep
from os.path import exists
from typing import Tuple, Union
import math
import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

TEXT_COLOR = (255, 0, 0)  # red
# Create an FaceDetector object.
dir = os.getcwd()

ultralytics.checks()
# load models
attention_model = ultralytics.YOLO(dir + '/models/best.pt')
# face_detector
base_options = python.BaseOptions(
    model_asset_path=f"{dir}/models/detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


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
    result_str = ''
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
            result_str = f"{(attentive_count / len(classes_detected)) * 100}% attentive: {(drowsy_count / len(classes_detected)) * 100}% drowsy"

    return result_str, attentive_count, drowsy_count


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
    attentive_count_all = 0
    drowsy_count_all = 0

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
        box_region = annotated_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        # TODO: comment cv2 image show function
        # cv2.imshow('detected',box_region)
        # pass box_region to your attention detection model
        result_string, attentive_count, drowsy_count = apply_attention_detection_model(
            box_region)
        attentive_count_all += attentive_count
        drowsy_count_all += drowsy_count
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
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image, attentive_count_all, drowsy_count_all

    # to test uploaded data


def apply_face_detection_inference(image):
    global detector
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # Detect faces in the input image.
    detection_result = detector.detect(mp_image)
    print(detection_result)

    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    # pass to visualize function that takes in image and detection result and then annotated image
    annotated_image, attentive_count_all, drowsy_count_all = visualize(
        image_copy, detection_result)

    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return rgb_annotated_image, attentive_count_all, drowsy_count_all


def object_detection_module(in_path):
    # get video name
    name = in_path.split('/')[-1].split('.')[0]
    input_video = cv2.VideoCapture(in_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(3))
    frame_height = int(input_video.get(4))
    # output video
    output_file_name = name + '-out.mp4'
    output_path = os.path.join(
        os.getcwd(), 'public', 'videos', output_file_name)
    print(output_path, fps, frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height))
    # read video frames
    a_count_all = 0
    d_count_all = 0
    while True:
        flag, frame = input_video.read()
        if flag:
            # The frame is ready and already captured
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_out, a_count, d_count = apply_face_detection_inference(
                frame_rgb)
            a_count_all += a_count
            d_count_all += d_count
            #   cv2.imshow('Video', frame_out)
            video_out.write(frame_out)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            break
    input_video.release()
    video_out.release()
    return output_file_name, a_count_all, d_count_all


"""
    ************************************
    *                                  *
    *         SERVER CODE SECTION      *
    *                                  *
    ************************************
"""


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'public/videos')
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def create_video():
    try:
        if 'video' not in request.files:
            return jsonify({
                'status': "error",
                'message': 'No video'
            }), 400

        video = request.files['video']

        if video.filename == '':
            return jsonify({
                'status': "error",
                'message': 'No selected video'
            }), 400

        if video and allowed_file(video.filename):
            timestamp = str(int(time.time()))
            filename = timestamp + '_' + video.filename

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath, 'filepath')
            with open(filepath, 'wb') as f:
                f.write(video.read())

            output_filename, attentive_count, drowsy_count = object_detection_module(
                filepath)
            print(output_filename, attentive_count, drowsy_count)

            return jsonify({
                'status': "success",
                'data': {
                    'result': filename,
                    'sum_inference': 1000,
                    'engaged_percent': 1000,
                    'boredom_percent': 80,
                    'confusion_percent': 90,
                    'frustration_percent': 110,
                    'distracted_percent': 600,
                    'attentive_count': attentive_count,
                    'drowsy_count': drowsy_count,
                    'total_count': attentive_count + drowsy_count
                },
                'message': 'Video uploaded successfully'
            }), 201
        else:
            return jsonify({
                'status': "error",
                'message': 'You can only upload a video in mp4 format'
            }), 500
    except Exception as e:
        traceback.print_exc()
        print(e)
        return jsonify({
            'status': "error",
            'message': 'Unable to upload video'
        }), 500


@app.route('/stream/<filename>', methods=['GET'])
def download_video(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)
        if os.path.exists(filepath):
            def generate():
                with open(filepath, 'rb') as f:
                    while True:
                        # adjust the chunk size as needed
                        chunk = f.read(1024)
                        if not chunk:
                            break
                        yield chunk

            return Response(generate(), mimetype='video/mp4')
        else:
            return jsonify({
                'status': 0,
                'message': 'File not found'
            }), 404
    except Exception as e:
        print(e)
        return jsonify({
            'status': 0,
            'message': 'Error while streaming file'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)
