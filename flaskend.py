from flask import Flask, render_template, request
import tester
from enum import Enum


class NoInput(Exception):
    # Raised when neural network failed to detect face/hand
    pass


class InputNames(str, Enum):
    Mode = "Mode",
    Camera_ID = "Camera ID",
    Distraction_Time = "Distraction Time",
    Detection_Confidence = "Detection Confidence",
    Tracking_Confidence = "Tracking Confidence"


app = Flask(__name__, template_folder='templates')


def detect_concentration(mode, camera_id, distraction_time, detection_confidence, tracking_confidence):
    concentration_detection = tester.ConcentrationDetection(mode, camera_id, distraction_time, detection_confidence,
                                                            tracking_confidence)
    concentration_detection.run()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['GET', "POST"])
def run():
    input_values = {
        InputNames.Mode: 0,
        InputNames.Camera_ID: 0,
        InputNames.Distraction_Time: 5,
        InputNames.Detection_Confidence: 0.5,
        InputNames.Tracking_Confidence: 0.5
    }
    if request.method == 'POST':
        for input_key in input_values:
            try:
                if input_key in ['Mode', 'Camera ID']:
                    input_values[input_key] = int(request.form[input_key])
                else:
                    input_values[input_key] = float(request.form[input_key])
            except NoInput:
                pass

        detect_concentration(input_values[InputNames.Mode], input_values[InputNames.Camera_ID],
                             input_values[InputNames.Distraction_Time],
                             input_values[InputNames.Detection_Confidence],
                             input_values[InputNames.Tracking_Confidence])
    return render_template('configuration.html')


@app.route('/info')
def info():
    return render_template('info.html')


if __name__ == "__main__":
    app.run(debug=True)
