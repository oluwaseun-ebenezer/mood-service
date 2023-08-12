from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import os
import time

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

            with open(filepath, 'wb') as f:
                f.write(video.read())

            # model time for video analysis
            time.sleep(10)

            # analyse video

            return jsonify({
                'status': "success",
                'data': {
                    'result': filename,
                    'sum_inference': 1000,
                    'engaged_percent': 1000,
                    'boredom_percent': 80,
                    'confusion_percent': 90,
                    'frustration_percent': 110,
                    'distracted_percent': 600
                },
                'message': 'Video uploaded successfully'
            }), 201
        else:
            return jsonify({
                'status': "error",
                'message': 'You can only upload a video in mp4 format'
            }), 500
    except Exception as e:
        print(e)
        return jsonify({
            'status': "error",
            'message': 'Unable to upload video'
        }), 500


@app.route('/stream/<filename>', methods=['GET'])
def download_video(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

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
