# Creado por Lucy
# Fecha: 05/07/2023

from flask import Flask, render_template, Response, request, redirect, url_for
import api.panorama_capture

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def gen(cam_function):
    while True:
        frame = cam_function()
        if frame is None:
            continue
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/panorama/capture", methods=["POST"])
def panorama_capture():
    num_images = int(request.form.get("num_images"))
    panorama_capture.capture_images(num_images)
    return redirect(url_for("index"))

@app.route("/object_tracking")
def object_tracking_stream():
    from api.object_tracking_realtime import process_object_tracking
    return Response(gen(process_object_tracking), content_type="multipart/x-mixed-replace; boundary=frame")

@app.route("/pose_estimation")
def pose_estimation_stream():
    from api.pose_estimation_realtime import process_pose_estimation
    return Response(gen(process_pose_estimation), content_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
