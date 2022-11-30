from flask import Flask, render_template, request, Response
import tester


class NoInput(Exception):
    # Raised when neural network failed to detect face/hand
    pass


class InputValues:
    def __init__(self):
        self.values = {
            tester.InputNames.Mode: 1,
            tester.InputNames.Camera_ID: 0,
            tester.InputNames.Distraction_Time: 5,
            tester.InputNames.Detection_Confidence: 0.5,
            tester.InputNames.Tracking_Confidence: 0.5
        }
        self.frame = None


input_values = InputValues()
concentration_detection = tester.ConcentrationDetection()
app = Flask(__name__, template_folder='templates')


def configure_concentration_detection(concentration_detection):

    while True:
        if concentration_detection.mode == tester.Modes.SHOWCASE:
            concentration_detection.showcase()
        else:
            concentration_detection.run()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + concentration_detection.frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/configuration', methods=['GET', "POST"])
def run():
    global input_values
    if request.method == 'POST':
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


@app.route('/video_feed')
def video_feed():
    global input_values, concentration_detection
    concentration_detection.set_input_data(input_values)

    return Response(configure_concentration_detection(concentration_detection), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/showcase', methods=['GET', "POST"])
def showcase():
    global input_values
    if request.method == 'POST':
        input_values.values[tester.InputNames.Camera_ID], input_values.values[tester.InputNames.Mode] \
            = int(request.form[tester.InputNames.Camera_ID]), 3
    return render_template('showcase.html')


if __name__ == "__main__":
    app.run(debug=True)
