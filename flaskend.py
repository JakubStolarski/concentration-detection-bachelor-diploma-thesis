from flask import Flask, render_template
import tester

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run')
def run():
    concentration_detection = tester.ConcentrationDetection(1, 1)
    concentration_detection.run()


if __name__ == "__main__":
    app.run(debug=True)
