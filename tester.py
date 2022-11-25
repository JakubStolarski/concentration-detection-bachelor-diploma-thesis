import cv2
from enum import IntEnum
import easygui as e
import mediapipe as mp
import numpy as np
import time


class DetectionError(Exception):
    # Raised when neural network failed to detect face/hand
    pass


class Modes(IntEnum):
    CALIBRATION = 0
    BASIC = 1
    VIDEO_ANALYSIS = 2
    SHOWCASE = 4


class Landmarks(IntEnum):
    NOSE_TIP = 1
    RIGHT_EYE = 33
    RIGHT_MOUTH = 61
    CHIN = 199
    LEFT_EYE = 263
    LEFT_MOUTH = 291


class Hand(IntEnum):
    WRIST = 0
    THUMB_BASE = 1
    THUMB_MID_LOW = 2  # proximal phalanx of thumb
    THUMB_MID_HIGH = 3  # distal phalanx of thumb
    THUMB_TIP = 4
    INDEX_MID_LOW = 6  # middle phalanx of index finger


class ConcentrationDetection:
    def __init__(self, mode, camera_id=0, distraction_tolerance=5, detection_confidence=0.5, tracking_confidence=0.5):
        self._activate_mp_solutions(detection_confidence, tracking_confidence)
        self._set_initial_boundaries()
        self.starter_time = time.time()
        self.alarm_flag = False
        self.mode = mode
        self.camera_id = camera_id
        self.distraction_tolerance = distraction_tolerance
        self.frame = None
        self.run_initialized = False
        self.curr_alarm_time = self.start_alarm_time = None
        self.bounded = False
        self.bounds = 0

    def _activate_mp_solutions(self, detection_confidence, tracking_confidence):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)

    def _set_initial_boundaries(self):
        initial_boundaries_limit = np.array([[100], [-100]])
        boundaries_shape = np.ones((2, 3), dtype=float)
        # workspace structure is an array of two arrays, first of which consist of lower
        self.bound_workspace = np.multiply(boundaries_shape, initial_boundaries_limit)
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def orientation(coordinate_landmark_0, coordinate_landmark_9):
        x0 = coordinate_landmark_0[0]
        y0 = coordinate_landmark_0[1]

        x9 = coordinate_landmark_9[0]
        y9 = coordinate_landmark_9[1]

        if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
            m = np.inf()
        else:
            m = abs((y9 - y0) / (x9 - x0))

        if 0 <= m <= 1:
            if x9 > x0:
                return "Right"
            else:
                return "Left"
        if m > 1:
            if y9 < y0:  # since, y decreases upwards
                return "Up"
            else:
                return "Down"

    @staticmethod
    def give_coordinate(x_or_y, landmark, results):
        coordinate = 0 if x_or_y == "x" else 1
        return float(
            str(results.multi_hand_landmarks[-1].landmark[int(landmark)]).split('\n')[coordinate].split(" ")[1])

    def _finger(self, landmark, results, width, height):
        # is z="finger, it returns which finger is closed. If z="true coordinate", it returns the true coordinates
        if results.multi_hand_landmarks is not None:
            try:
                plandmark_x = self.give_coordinate("x", landmark, results)
                plandmark_y = self.give_coordinate("y", landmark, results)
                return int(width * plandmark_x), int(height * plandmark_y)

            except DetectionError:
                pass

    @staticmethod
    def _get_translation_and_rotation(frame, results_face):
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [Landmarks.NOSE_TIP, Landmarks.RIGHT_EYE, Landmarks.RIGHT_MOUTH,
                               Landmarks.CHIN, Landmarks.LEFT_EYE, Landmarks.LEFT_MOUTH]:
                        if idx == Landmarks.NOSE_TIP:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                vid, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the rotation degrees
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                print(x, y, z)

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                 dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                return [p1, p2, x, y, z]

    def _frame_operations(self):
        is_ok = x = y = z = None
        # Flip the frame horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        self.frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance
        self.frame.flags.writeable = False

        # Get the result
        results_face = self.face_mesh.process(self.frame)
        results_hand = self.hands.process(self.frame)

        # To improve performance
        self.frame.flags.writeable = True

        # Convert the color space from RGB to BGR
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        if results_face.multi_face_landmarks:
            [p1, p2, x, y, z] = self._get_translation_and_rotation(self.frame, results_face)
            if self.mode == Modes.SHOWCASE:
                cv2.line(self.frame, p1, p2, (255, 0, 0), 2)

        if results_hand.multi_hand_landmarks:
            for finger_landmark in [Hand.THUMB_BASE, Hand.THUMB_MID_LOW, Hand.THUMB_MID_HIGH, Hand.INDEX_MID_LOW]:
                is_ok = True
                thumb_tip = self._finger(4, results_hand, self.frame.shape[1], self.frame.shape[0])
                section_landmark = self._finger(finger_landmark, results_hand, self.frame.shape[1], self.frame.shape[0])
                if thumb_tip[1] > section_landmark[1]:
                    is_ok = False
                    break

        return is_ok, results_hand, results_face, x, y, z

    def _initialize(self):
        self.cap = cv2.VideoCapture(self.camera_id)

    def run(self):
        if not self.run_initialized:
            self._initialize()
            self.run_initialized = True

        vid, self.frame = self.cap.read()
        is_ok, results_hand, results_face, x, y, z = self._frame_operations()

        if is_ok and self.mode in [Modes.CALIBRATION, Modes.SHOWCASE]:
            cv2.putText(self.frame, "Okay!!", (int(self.frame.shape[1]*0.7), int(self.frame.shape[0]*0.85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            try:
                workspace = [[x, y, z]]
                for point in workspace:
                    for elem_num, coordinate in enumerate(point):
                        if coordinate < self.bound_workspace[0][elem_num]:
                            self.bound_workspace[0][elem_num] = coordinate
                            self.bounded = True
                        if coordinate > self.bound_workspace[1][elem_num]:
                            self.bound_workspace[1][elem_num] = coordinate
                            self.bounded = True
                        if self.bounded:
                            self.bounds += 1
                            self.bounded = False
                if self.bounds >= 4:
                    self.alarm_flag = True
                    if x in range(int(self.bound_workspace[0][0]), int(self.bound_workspace[1][0])):
                        if y in range(int(self.bound_workspace[0][1]), int(self.bound_workspace[1][1])):
                            if z in range(int(self.bound_workspace[0][2]), int(self.bound_workspace[1][2])):
                                self.alarm_flag = False
            except DetectionError:
                workspace = [[x, y, z]]
                self.bounds = 1

        if results_hand.multi_hand_landmarks and self.mode == Modes.SHOWCASE:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if self.bounds > 3:
            alarm_flag = True
            if results_face.multi_face_landmarks:
                if int(x) in range(int(self.bound_workspace[0][0]), int(self.bound_workspace[1][0])):
                    # todo add funtion for checking
                    if int(y) in range(int(self.bound_workspace[0][1]), int(self.bound_workspace[1][1])):
                        if self.bound_workspace[0][2] < z < self.bound_workspace[1][2]:
                            alarm_flag = False

            if alarm_flag:
                if not self.curr_alarm_time:
                    self.curr_alarm_time = time.time()
                    self.start_alarm_time = time.time()
                else:
                    self.curr_alarm_time = time.time()
                    if self.curr_alarm_time - self.start_alarm_time > self.distraction_tolerance:
                        e.msgbox("An error has occured! :(", "Error")
                        cv2.putText(self.frame, "Alarm!",
                                    (int(self.frame.shape[1] / 2), int(self.frame.shape[0] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            else:
                self.curr_alarm_time = self.start_alarm_time = None

        if self.mode == Modes.SHOWCASE:
            running_time = "--- %s seconds ---" % (time.time() - self.starter_time)
            cv2.putText(self.frame, running_time, (int(self.frame.shape[1] - 200), int(self.frame.shape[0] - 200)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # cv2.imshow('Head Pose Estimation', self.frame)
        ret, jpeg = cv2.imencode('.jpg', self.frame)

        self.frame = jpeg.tobytes()


if __name__ == "__main__":
    concentration_detection = ConcentrationDetection(1, 1)
    while True:
        concentration_detection.run()

