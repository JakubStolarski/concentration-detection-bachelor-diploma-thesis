import cv2
import mediapipe as mp
import numpy as np
import time
from enum import IntEnum


class DetectionError(Exception):
    # Raised when neural network failed to detect face/hand
    pass


class Modes(IntEnum):
    CALIBRATION = 0
    BASIC = 1
    VIDEO_ANALYSIS = 2


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
    def __init__(self, mode, camera_id=0, detection_confidence=0.5, tracking_confidence=0.5):
        self._activate_mp_solutions(detection_confidence, tracking_confidence)
        self._set_initial_boundaries()

        self.starter_time = time.time()
        self.alarm_flag = False
        self.mode = mode
        self.camera_id = camera_id

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
        p1 = p2 = x = y = z = None
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [Landmarks.NOSE_TIP, Landmarks.RIGHT_EYE, Landmarks.RIGHT_MOUTH,
                               Landmarks.CHIN, Landmarks.LEFT_EYE, Landmarks.LEFT_MOUTH]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
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

                # Get the y rotation degree
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

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        curr_alarm_time = None
        bounds = 0
        while cap.isOpened():
            vid, frame = cap.read()
            # Flip the frame horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            frame.flags.writeable = False

            # Get the result
            results_face = self.face_mesh.process(frame)
            results_hand = self.hands.process(frame)

            # To improve performance
            frame.flags.writeable = True

            # Convert the color space from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results_face.multi_face_landmarks:
                [p1, p2, x, y, z] = self._get_translation_and_rotation(frame, results_face)
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            if results_hand.multi_hand_landmarks:
                for finger_landmark in [Hand.THUMB_BASE, Hand.THUMB_MID_LOW, Hand.THUMB_MID_HIGH, Hand.INDEX_MID_LOW]:
                    accepted = True
                    thumb_tip = self._finger(4, results_hand, frame.shape[1], frame.shape[0])
                    section_landmark = self._finger(finger_landmark, results_hand, frame.shape[1], frame.shape[0])
                    if thumb_tip[1] > section_landmark[1]:  # todo self describing code
                        accepted = False
                        break

                if accepted:
                    cv2.putText(frame, "Okay!!", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
                    try:
                        workspace = [[x, y, z]]
                        for point in workspace:
                            for elem_num, coordinate in enumerate(point):
                                if coordinate < self.bound_workspace[0][elem_num]:
                                    self.bound_workspace[0][elem_num] = coordinate
                                if coordinate > self.bound_workspace[1][elem_num]:
                                    self.bound_workspace[1][elem_num] = coordinate
                        bounds += 1
                        if bounds >= 4:
                            self.alarm_flag = True
                            if x in range(int(self.bound_workspace[0][0]), int(self.bound_workspace[1][0])):
                                if y in range(int(self.bound_workspace[0][1]), int(self.bound_workspace[1][1])):
                                    if z in range(int(self.bound_workspace[0][2]), int(self.bound_workspace[1][2])):
                                        self.alarm_flag = False
                    except DetectionError:
                        workspace = [[x, y, z]]
                        bounds = 1
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            if bounds > 3:
                alarm_flag = True
                if int(x) in range(int(self.bound_workspace[0][0]),
                                   int(self.bound_workspace[1][0])):  # todo add funtion for checking
                    if int(y) in range(int(self.bound_workspace[0][1]), int(self.bound_workspace[1][1])):
                        if self.bound_workspace[0][2] < z < self.bound_workspace[1][2]:
                            alarm_flag = False

                if alarm_flag:
                    if not curr_alarm_time:
                        curr_alarm_time = time.time()
                        start_alarm_time = time.time()
                    else:
                        curr_alarm_time = time.time()
                        if curr_alarm_time - start_alarm_time > 5:
                            cv2.putText(frame, "Alarm!", (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
                else:
                    curr_alarm_time = start_alarm_time = None

            cv2.imshow('Head Pose Estimation', frame)
            print("--- %s seconds ---" % (time.time() - self.starter_time))
            if cv2.waitKey(5) & 0xFF == 27:
                print(workspace)
                break
        cap.release()


if __name__ == "__main__":
    concentration_detection = ConcentrationDetection(1, 1)
    concentration_detection.run()
