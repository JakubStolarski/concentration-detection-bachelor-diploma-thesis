import cv2
import mediapipe as mp
import numpy as np
import time


class DetectionError(Exception):
    # Raised when neural network failed to detect face/hand
    pass


def give_coordinate(x_or_y, landmark, results):
    coordinate = 0 if x_or_y == "x" else 1
    return float(str(results.multi_hand_landmarks[-1].landmark[int(landmark)]).split('\n')[coordinate].split(" ")[1])


def finger(landmark, results, width, height):
    # is z="finger, it returns which finger is closed. If z="true coordinate", it returns the true coordinates
    if results.multi_hand_landmarks is not None:
        try:
            plandmark_x = give_coordinate("x", landmark, results)
            plandmark_y = give_coordinate("y", landmark, results)
            return int(width * plandmark_x), int(height * plandmark_y)

        except DetectionError:
            pass


def orientation(coordinate_landmark_0, coordinate_landmark_9):
    x0 = coordinate_landmark_0[0]
    y0 = coordinate_landmark_0[1]

    x9 = coordinate_landmark_9[0]
    y9 = coordinate_landmark_9[1]

    if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
        m = 1000000000
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


if __name__ == "__main__":

    starter_time = time.time()
    curr_alarm_time = None
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    [xmin, ymin, zmin] = [100, 100, 100]
    [xmax, ymax, zmax] = [-100, -100, -100]
    bound_workspace = [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    alarm_flag = False
    workspace_bounds = 0
    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results_face = face_mesh.process(image)
        results_hand = hands.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        accepted = False
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
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
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

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
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

        if results_hand.multi_hand_landmarks:
            for finger_landmark in [1, 2, 3, 6]:
                accepted = True
                a = finger(4, results_hand, image.shape[1], image.shape[0])
                b = finger(finger_landmark, results_hand, image.shape[1], image.shape[0])
                if a[1] > b[1]:  # todo self describing code
                    accepted = False
                    break

            if accepted:
                cv2.putText(image, "Okay!!", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
                try:
                    workspace = [[x, y, z]]
                    for point in workspace:
                        for elem_num, coordinate in enumerate(point):
                            if coordinate < bound_workspace[0][elem_num]:
                                bound_workspace[0][elem_num] = coordinate
                            if coordinate > bound_workspace[1][elem_num]:
                                bound_workspace[1][elem_num] = coordinate
                    workspace_bounds += 1
                    if workspace_bounds >= 4:
                        alarm_flag = True
                        if x in range(int(bound_workspace[0][0]), int(bound_workspace[1][0])):
                            if y in range(int(bound_workspace[0][1]), int(bound_workspace[1][1])):
                                if z in range(int(bound_workspace[0][2]), int(bound_workspace[1][2])):
                                    alarm_flag = False
                except DetectionError:
                    workspace = [[x, y, z]]
                    workspace_bounds = 1
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if workspace_bounds > 3:
            alarm_flag = True
            if int(x) in range(int(bound_workspace[0][0]), int(bound_workspace[1][0])):  # todo add funtion for checking
                if int(y) in range(int(bound_workspace[0][1]), int(bound_workspace[1][1])):
                    if bound_workspace[0][2] < z < bound_workspace[1][2]:
                        alarm_flag = False

            if alarm_flag:
                if not curr_alarm_time:
                    curr_alarm_time = time.time()
                    start_alarm_time = time.time()
                else:
                    curr_alarm_time = time.time()
                    if curr_alarm_time-start_alarm_time > 5:
                        cv2.putText(image, "Alarm!", (int(image.shape[1] / 2), int(image.shape[0] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            else:
                curr_alarm_time = start_alarm_time = None

        if cv2.waitKey(5) & 0xFF == 27:
            print(workspace)
            break
    cap.release()
