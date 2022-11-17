from flask import Flask, render_template
import win32api
from logging.config import dictConfig
import cv2 as cv
import numpy as np
import mediapipe as mp

# dictConfig({
#     'version': 1,
#     'formatters': {'default': {
#         'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
#     }},
#     'handlers': {'wsgi': {
#         'class': 'logging.StreamHandler',
#         'stream': 'ext://flask.logging.wsgi_errors_stream',
#         'formatter': 'default'
#     }},
#     'root': {
#         'level': 'INFO',
#         'handlers': ['wsgi']
#     }
# })

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run')
def run():
    mp_face_mesh = mp.solutions.face_mesh
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # right eyes indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    NOSE = [45, 4, 275, 274, 1, 44, 125, 19, 354]  # [20, 242, 141, 24, 370, 462, 250, 458, 461, 354, 19, 125, 241, 238]
    CHIN = [152, 175, 199, 200]
    # [171, 175, 396, 277, 152, 148]
    # #[100, 269, 262, 428, 199, 208, 32, 140, 171, 175, 306, 369, 100, 277, 152, 148, 176, 140, 32]
    LEFT_EAR = [356, 454, 323, 361]
    RIGHT_EAR = [127, 234, 93, 132]
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                # print(results_face.multi_face_landmarks[0].landmark)
                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                     results.multi_face_landmarks[0].landmark])
                # print(mesh_points.shape)
                # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
                # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                (n_cx, n_cy), n_radius = cv.minEnclosingCircle(mesh_points[NOSE])
                (c_cx, c_cy), c_radius = cv.minEnclosingCircle(mesh_points[CHIN])
                (le_cx, le_cy), le_radius = cv.minEnclosingCircle(mesh_points[LEFT_EAR])
                (re_cx, re_cy), re_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EAR])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                center_nose = np.array([n_cx, n_cy], dtype=np.int32)
                center_chin = np.array([c_cx, c_cy], dtype=np.int32)
                center_r_ear = np.array([re_cx, re_cy], dtype=np.int32)
                center_l_ear = np.array([le_cx, le_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_nose, int(n_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_chin, int(c_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_l_ear, int(le_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_r_ear, int(re_radius), (255, 0, 255), 1, cv.LINE_AA)

            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    #file = open(r'C:\Users\kubas\OneDrive\Pulpit\do_szkoły\AAA_Projekt\concentration-detection-bachelor-diploma-thesis\face_landmark_detecion_and_showcase.py', 'r').read()


if __name__ == "__main__":
    app.run(debug=True)
