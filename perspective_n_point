import cv2
import dlib
import numpy as np
from imutils import face_utils

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])


def load_landmark_model(model_path):
    return dlib.shape_predictor(model_path)


def calibrate_camera(image):
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    cam_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    print("Camera Matrix :\n {0}".format(cam_matrix))
    dst_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    return cam_matrix, dst_coeffs


def solve_head_pose(landmarks):
    image_pts = np.float32([landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48],
                            landmarks[54]])

    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_pts, camera_matrix, dist_coeffs)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    # calc euler angle
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    return nose_end_point2D, euler_angles


def cvt_x_to_deg(euler_x):
    degree = 180 - abs(euler_x)
    if euler_x > 0:
        return degree
    else:
        return -degree


face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
predictor = load_landmark_model(model_path=face_landmark_path)
# video_path = '/run/media/root/Data/E-KYC_POSE/static/images/video1599561278.666116.mp4'
video_path = 0
cap = cv2.VideoCapture(video_path)
ref_ret, ref_frame = cap.read()
# ref_frame = cv2.rotate(ref_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
camera_matrix, dist_coeffs = calibrate_camera(image=ref_frame)
detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ret:
        # frame = cv2.medianBlur(frame, 5)
        face_rects = detector(frame, 0)

        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            nose_end_point, euler_angle = solve_head_pose(shape)
            # euler_angle = solve_head_pose(shape)

            p1 = (int(shape[30][0]), int(shape[30][1]))
            p2 = (int(nose_end_point[0][0][0]), int(nose_end_point[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 4)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 128, 128), -1)

            cv2.putText(frame, "X: " + "{:7.2f}".format(cvt_x_to_deg(euler_angle[0, 0])), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("camera", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

