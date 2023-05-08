# Creado por Lucy
# Fecha: 05/07/2023

from flask import Flask, render_template, Response, url_for
from api.panorama_capture import capture_images
from api.object_tracking_realtime import process_object_tracking
from api.pose_estimation_realtime import process_pose_estimation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_panorama(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_tracking():
    for frame in process_object_tracking('MOSSE', 'MEDIANFLOW'):#('MOSSE', 'CSRT', 'BOOSTING', 'MEDIANFLOW'):  # Puedes agregar más trackers aquí
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_estimation():
    for frame in process_pose_estimation():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/panorama_stream')
def panorama_stream():
    return Response(gen_panorama(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_tracking_stream')
def object_tracking_stream():
    return Response(gen_tracking(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_estimation_stream')
def pose_estimation_stream():
    return Response(gen_estimation(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/panorama')
def panorama():
    return render_template('panorama.html')

@app.route('/object_tracking')
def object_tracking():
    return render_template('object_tracking.html')

@app.route('/pose_estimation')
def pose_estimation():
    return render_template('pose_estimation.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
