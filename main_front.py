import flask
from flask import Flask, redirect, render_template, request, Response, url_for
import pandas as pd
import concentration_detection


class NoInput(Exception):
    # Raised when neural network failed to detect face/hand
    pass


class InputValues:
    def __init__(self):
        self.values = {
            concentration_detection.InputNames.Mode: None,
            concentration_detection.InputNames.Camera_ID: 0,
            concentration_detection.InputNames.Distraction_Time: 5,
            concentration_detection.InputNames.Detection_Confidence: 0.5,
            concentration_detection.InputNames.Tracking_Confidence: 0.5
        }
        self.frame = None
        self.workspace = None


input_values = concentration_detector = None


def base_state():
    global input_values, concentration_detector
    input_values = InputValues()
    concentration_detector = concentration_detection.ConcentrationDetection()


app = Flask(__name__, template_folder='templates')


def configure_concentration_detection(concentration_detector):
    while True:
        if concentration_detector.mode is not None:
            if concentration_detector.mode == concentration_detection.Modes.SHOWCASE:
                concentration_detector.showcase()
            else:
                if input_values.workspace is not None:
                    concentration_detector.read_workspace(input_values.workspace)
                concentration_detector.run()
        else:
            pass
        # if no frame was captured return a single byte
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'
               + concentration_detector.frame + b'\r\n\r\n') if concentration_detector.frame else bytes()


@app.route('/')
def index():
    base_state()
    return render_template('index.html')


@app.route('/configuration', methods=['GET', "POST"])
def run():
    global input_values
    if request.method == 'POST':
        data = request.files['workspace']
        if not data.filename == '':
            df = pd.read_json(data)
            input_values.workspace = df.values

        for input_key in input_values.values:
            try:
                if input_key in ['Mode', 'Camera ID']:
                    input_values.values[input_key] = int(request.form[input_key])
                else:
                    input_values.values[input_key] = float(request.form[input_key])
            except NoInput:
                pass

    return render_template('configuration.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/showcase', methods=['GET', "POST"])
def showcase():
    global input_values
    if request.method == 'POST':
        input_values.values[concentration_detection.InputNames.Camera_ID], \
        input_values.values[concentration_detection.InputNames.Mode] \
            = int(request.form[concentration_detection.InputNames.Camera_ID]), 3
    return render_template('showcase.html')


@app.route('/video_feed')
def video_feed():
    global input_values, concentration_detector
    concentration_detector.set_input_data(input_values)

    return Response(configure_concentration_detection(concentration_detector), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_and_quit')
def save_and_quit():
    concentration_detector.save_workspace()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
