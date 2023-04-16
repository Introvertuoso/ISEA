import argparse
import math
import pickle
import sys
import os
import cv2
import numpy as np


# Setting OpenPose parameters
def set_params():
    params = dict()
    # params["model_folder"] = "../../../models/"
    # TODO: Introduce a relative to absolute path solver helper function here too
    params["model_folder"] = "C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\fall_detection" \
                             "\\openpose\\models\\"
    params["disable_blending"] = False
    params["net_resolution"] = "-1x256"
    return params


def correct_coordinates(keypoints):
    for keypoint in keypoints:
        keypoint[1] = -keypoint[1]
    return keypoints


# Checks if the pose estimators are able to pinpoint the joint/keypoint in question.
# This will later also check the confidence of the estimation and filter based on it as well.
def are_keypoints_valid(person):
    head_x, head_y = person[0][0], person[0][1]
    neck_x, neck_y = person[1][0], person[1][1]
    center_x, center_y = person[8][0], person[8][1]
    r_knee_x, r_knee_y = person[10][0], person[10][1]
    l_knee_x, l_knee_y = person[13][0], person[13][1]
    if 0 not in (head_x, neck_x, center_x, r_knee_x, l_knee_x,
                 head_y, neck_y, center_y, r_knee_y, l_knee_y):
        return True
    else:
        return False


def euclidean_distance(neck, center):
    return (((center[0] - neck[0]) ** 2) + ((center[1] - neck[1]) ** 2)) ** 0.5


def angle(head, center):
    return np.arctan(abs((head[1] - center[1]) / (head[0] - center[0])))


def bounding_box(head, r_knee, l_knee):
    max_x = max([head[0], r_knee[0], l_knee[0]])
    min_x = min([head[0], r_knee[0], l_knee[0]])
    max_y = max([head[1], r_knee[1], l_knee[1]])
    min_y = min([head[1], r_knee[1], l_knee[1]])
    return max_x, min_x, max_y, min_y


def ratio(head, r_knee, l_knee):
    max_x, min_x, max_y, min_y = bounding_box(head, r_knee, l_knee)
    return (max_x - min_x) / (max_y - min_y)


def angle_3p(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    ba = a - b
    bc = c - b
    denominator = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator == 0:
        denominator = small_number
    cosine_angle = np.dot(ba, bc) / denominator
    ang = np.arccos(cosine_angle)
    return np.degrees(ang)


def angular_speed(delta_time, old_pos, new_pos, old_pivot, new_pivot):
    old_angle = angle_3p(old_pos, old_pivot, np.array([old_pivot[0] + 1, old_pivot[1]]))
    new_angle = angle_3p(new_pos, new_pivot, np.array([new_pivot[0] + 1, new_pivot[1]]))
    return abs(old_angle - new_angle) / delta_time


def angular_acceleration(delta_time, old_pos, new_pos, next_pos, old_pivot, new_pivot, next_pivot):
    speed1 = angular_speed(delta_time, old_pos, new_pos, old_pivot, new_pivot)
    speed2 = angular_speed(delta_time, new_pos, next_pos, new_pivot, next_pivot)
    return abs(speed1 - speed2) / delta_time


def calculate_feature_vector(evolution):
    e_rot = []
    q = []
    t_minus_2_head = t_minus_2_neck = t_minus_2_center = t_minus_2_r_knee = t_minus_2_l_knee = np.array((0, 0, 0))
    t_minus_1_head = t_minus_1_neck = t_minus_1_center = t_minus_1_r_knee = t_minus_1_l_knee = np.array((0, 0, 0))
    t_head = t_neck = t_center = t_r_knee = t_l_knee = np.array((0, 0, 0))
    t_minus_1_mid_point = np.array((0, 0, 0))
    t_mid_point = np.array((0, 0, 0))
    time_to_process_frame = 1 / fps
    for t in range(len(evolution)):
        t_minus_2_head, t_minus_2_neck, t_minus_2_center, t_minus_2_r_knee, t_minus_2_l_knee = \
            t_minus_1_head, t_minus_1_neck, t_minus_1_center, t_minus_1_r_knee, t_minus_1_l_knee
        t_minus_1_head, t_minus_1_neck, t_minus_1_center, t_minus_1_r_knee, t_minus_1_l_knee = \
            t_head, t_neck, t_center, t_r_knee, t_l_knee
        t_head, t_neck, t_center, t_r_knee, t_l_knee = \
            evolution[t][0], evolution[t][1], evolution[t][8], evolution[t][10], evolution[t][13]
        t_head_rel, t_neck_rel, t_center_rel, t_r_knee_rel, t_l_knee_rel = \
            t_head - t_minus_1_head, t_head - t_minus_1_neck, t_neck - t_minus_1_center, \
            t_r_knee - t_minus_1_r_knee, t_l_knee - t_minus_1_l_knee
        t_minus_1_head_rel, t_minus_1_neck_rel, t_minus_1_center_rel, t_minus_1_r_knee_rel, t_minus_1_l_knee_rel = \
            t_minus_1_head - t_minus_2_head, t_minus_1_head - t_minus_2_neck, t_minus_1_neck - t_minus_2_center, \
            t_minus_1_r_knee - t_minus_2_r_knee, t_minus_1_l_knee - t_minus_2_l_knee
        t_mid_point = (t_head + t_neck) / 2
        t_minus_1_mid_point = t_mid_point
        t_minus_2_mid_point = t_minus_1_mid_point
        t_mid_point_rel = t_mid_point - t_minus_1_mid_point
        t_minus_1_mid_point_rel = t_minus_1_mid_point - t_minus_2_mid_point

        if t_minus_1_mid_point[2] is not 0:
            mid_point_e_rot = 1 * euclidean_distance(t_mid_point_rel, t_center_rel) ** 2 * angular_speed(
                time_to_process_frame, t_minus_1_mid_point_rel, t_mid_point_rel, t_minus_1_center_rel, t_center_rel
            ) ** 2
            r_knee_e_rot = 1 * euclidean_distance(t_r_knee_rel, t_center_rel) ** 2 * angular_speed(
                time_to_process_frame, t_minus_1_r_knee_rel, t_r_knee_rel, t_minus_1_center_rel, t_center_rel
            ) ** 2
            l_knee_e_rot = 1 * euclidean_distance(t_l_knee_rel, t_center_rel) ** 2 * angular_speed(
                time_to_process_frame, t_minus_1_l_knee_rel, t_l_knee_rel, t_minus_1_center_rel, t_center_rel
            ) ** 2
            e_rot.append((mid_point_e_rot + r_knee_e_rot + l_knee_e_rot) / 2)

        if t_minus_2_mid_point[2] is not 0:
            t_minus_2_mid_point = (t_minus_2_head + t_minus_2_neck) / 2
            denominator = (t_neck[0] - t_center[0])
            if denominator == 0:
                denominator = small_number
            m = (t_neck[1] - t_center[1]) / denominator
            if m == 0:
                m = small_number
            b = t_center[1] - m * t_center[0]
            y = t_head[1]
            x = math.floor((y - b) / m)
            theta_1 = angle_3p(
                t_head, t_neck, np.array([x, y])
            )
            theta_1_dot = angular_speed(
                time_to_process_frame, t_minus_1_head, t_head, t_minus_1_neck, t_neck
            )
            theta_1_dot_dot = angular_acceleration(
                time_to_process_frame, t_minus_2_head, t_minus_1_head, t_head,
                t_minus_2_neck, t_minus_1_neck, t_neck
            )
            theta_2 = angle_3p(
                t_neck, t_center, np.array([t_center[0], t_center[1] - 1])
            )
            theta_2_dot = angular_speed(
                time_to_process_frame, t_minus_1_neck, t_neck, t_minus_1_center, t_center
            )
            theta_2_dot_dot = angular_acceleration(
                time_to_process_frame, t_minus_2_neck, t_minus_1_neck, t_neck,
                t_minus_2_center, t_minus_1_center, t_center
            )
            d_1 = euclidean_distance(t_head, t_neck)
            d_2 = euclidean_distance(t_neck, t_center)
            g = 9.8
            q_1 = 1 * d_1 * theta_1_dot_dot ** 2 + (1 * d_1 ** 2 + 1 * d_1 * d_2 * np.cos(theta_1)) * theta_2_dot_dot + \
                  1 * d_1 * d_2 * np.sin(theta_1 * theta_2_dot ** 2) - 1 * g * d_2 * np.sin(theta_1 + theta_2)
            q_2 = (1 * d_1 ** 2 + 1 * d_1 * d_2 * np.cos(theta_1)) * theta_1_dot_dot + \
                  ((1 + 1) * d_2 ** 2 + 1 * d_1 ** 2 + 2 * 1 * d_1 * d_2 * np.cos(theta_1)) * theta_2_dot_dot - \
                  2 * 1 * d_1 * d_2 * np.sin(theta_1 * theta_1_dot * theta_2_dot) - 1 * d_1 * d_2 * np.sin(
                theta_2 * theta_1_dot ** 2) - (1 + 1) * g * d_2 * np.sin(theta_2) - \
                  1 * g * d_1 * np.sin(theta_1 + theta_2)
            q.append(q_1)
            q.append(q_2)
    vector = q + e_rot
    return vector


def ratio_angle_features(ev):
    vector = []
    for t in range(len(ev)):
        head, neck, center, r_knee, l_knee = \
            ev[t][0], ev[t][1], ev[t][8], ev[t][10], ev[t][13]
        vector.append(ratio(head, r_knee, l_knee))
        vector.append(angle(head, center))
    return vector


def setup_openpose():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    try:
        if sys.platform == "win32":
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            sys.path.append('../../python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this '
              'Python script in the right folder?')
        raise e
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    params = set_params()
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    return op, opWrapper


def fall_detection(current_keypoints, prev_keypoints, frames_elapsed, model):
    if not are_keypoints_valid(current_keypoints):
        if not are_keypoints_valid(prev_keypoints):
            return 0
        else:
            current_keypoints = prev_keypoints
    if len(buffer) == buffer_size:
        buffer.pop(0)
    buffer.append(current_keypoints)
    if frames_elapsed % step_size == 0 and len(buffer) == buffer_size:
        evolution = buffer
        features = calculate_feature_vector(evolution)
        features.extend(ratio_angle_features(evolution))
        features = np.array(features)
        features = features.reshape(-1, 1).transpose()
        return model.predict(features)[0]


buffer = []
buffer_size = 16
step_size = 8
fps = step_size
small_number = 0.000001


#  ONLY 30 FPS CAM
def start(conn):
    frames_elapsed = 0
    o_keypoints = None

    op, opWrapper = setup_openpose()
    opWrapper.start()

    # TODO: Introduce a relative to absolute path solver helper function here too
    with open('C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\fall_detection\\openpose'
              '\\build_windows\\examples\\user_code\\model-our_dataset+URFD.pkl', 'rb') as file:
        model = pickle.load(file)
        if model is None:
            raise FileNotFoundError

    while True:
        frame = conn.recv()
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        n_keypoints = datum.poseKeypoints
        # if n_keypoints is not None:
        if n_keypoints is not None:
            if len(n_keypoints) == 1:
                frames_elapsed += 1
                n_keypoints = correct_coordinates(n_keypoints[0])
                if o_keypoints is not None:
                    # conn.send(fall_detection(n_keypoints, o_keypoints, frames_elapsed, model))
                    pred = fall_detection(n_keypoints, o_keypoints, frames_elapsed, model)
                    if pred == 1:
                        # conn.send([0, 0, frame.shape[1], frame.shape[0]])
                        mxx, mnx, mxy, mny = bounding_box(n_keypoints[0], n_keypoints[10], n_keypoints[13])
                        conn.send([[int(abs(mnx)), int(abs(mny)), int(abs(mxx)), int(abs(mxy)), 4]])
                    else:
                        conn.send(None)
                else:
                    conn.send(None)
                o_keypoints = n_keypoints
            else:
                conn.send(None)
                frames_elapsed = 0
        else:
            conn.send(None)
            frames_elapsed = 0
        frame = datum.cvOutputData
        # if frame.shape[0] > 0:
        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) == 27:
            break
