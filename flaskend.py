from flask import Flask, render_template, request
import tester

app = Flask(__name__, template_folder='templates')


def detect_concentration(mode, camera_id, detection_confidence, tracking_confidence):
    concentration_detection = tester.ConcentrationDetection(mode, camera_id, detection_confidence, tracking_confidence)
    concentration_detection.run()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['GET', "POST"])
def run():
    if request.method == 'POST':
        mode = int(request.form['Mode'])
        camera_id = int(request.form['Camera ID'])
        detection_confidence = float(request.form['Detection Confidence'])
        tracking_confidence = float(request.form['Tracking Confidence'])
        if mode and camera_id and detection_confidence and tracking_confidence:
            detect_concentration(mode, camera_id, detection_confidence, tracking_confidence)
    return render_template('configuration.html')


@app.route('/info')
def info():
    return render_template('info.html')


if __name__ == "__main__":
    app.run(debug=True)
