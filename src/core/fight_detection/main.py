import collections
import math
import time

import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2


def get_mobilenet():
    mobilenet = MobileNetV2(weights="imagenet", include_top=True)
    mobile_net_transfer_layer = mobilenet.get_layer(mobilenet.layers[-2].name)
    mobilenet = tf.keras.models.Model(inputs=mobilenet.
                               input, outputs=mobile_net_transfer_layer.output)

    return mobilenet


def get_two_streams_model():
    return tf.keras.models.load_model(
        'C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\fight_detection\\model-best.h5')


def get_frames_framegrouping(frames):
    frame_groups = []
    for i in range(0, len(frames) - 2, 3):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        frame3 = frames[i + 2]
        oo = (0.3 * frame1[:, :, 0] + 0.59 * frame1[:, :, 1] + 0.11 * frame1[:, :, 2])
        uu = (0.3 * frame2[:, :, 0] + 0.59 * frame2[:, :, 1] + 0.11 * frame2[:, :, 2])
        bb = (0.3 * frame3[:, :, 0] + 0.59 * frame3[:, :, 1] + 0.11 * frame3[:, :, 2])

        kkl = np.array([oo, uu, bb])
        frame_group = np.moveaxis(kkl, 0, 2)

        frame_groups.append((frame_group / 255.).astype(np.float16))

    return np.array(frame_groups)


def get_frames_framegrouping_diff(frames):
    diff_frames = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        frame_diff = cv2.absdiff(frame2, frame1)
        diff_frames.append(frame_diff)

    diff_frames.append((np.zeros(frames[0].shape)).astype(np.float16))
    diff_frames = np.array(diff_frames)

    frame_groups = []

    for i in range(0, len(diff_frames) - 2, 3):
        frame1 = diff_frames[i]
        frame2 = diff_frames[i + 1]
        frame3 = diff_frames[i + 2]
        oo = (0.3 * frame1[:, :, 0] + 0.59 * frame1[:, :, 1] + 0.11 * frame1[:, :, 2])
        uu = (0.3 * frame2[:, :, 0] + 0.59 * frame2[:, :, 1] + 0.11 * frame2[:, :, 2])
        bb = (0.3 * frame3[:, :, 0] + 0.59 * frame3[:, :, 1] + 0.11 * frame3[:, :, 2])

        kkl = np.array([oo, uu, bb])
        frame_group = np.moveaxis(kkl, 0, 2)

        frame_groups.append((frame_group / 255.).astype(np.float16))

    return np.array(frame_groups)


def start(conn):
    frames = []
    frames_grouped_mobilenet_embeddings = collections.deque(maxlen=10)
    frames_grouped_diff_mobilenet_embeddings = collections.deque(maxlen=10)
    interval = 6
    frame_num = 0
    fps = 0
    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    mobilenet = get_mobilenet()
    two_streams_model = get_two_streams_model()

    #  one loop takes less than 15ms normally and 300ms every 5 seconds
    # is_Fight = False
    # alert_counter = 10
    while True:
        original = conn.recv()
        # original = cv2.GaussianBlur(original, (5, 5), 0)
        if frame_num % interval == 0:

            frame = cv2.resize(original, (224, 224), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames.append(frame)

        frame_num += 1

        if len(frames) == 3:
            frames_grouped = get_frames_framegrouping(frames)
            frames_grouped_diff = get_frames_framegrouping(frames)
            frames = []
            frames_grouped_mobilenet_embeddings.append(mobilenet.predict(frames_grouped)[0])
            frames_grouped_diff_mobilenet_embeddings.append(mobilenet.predict(frames_grouped_diff)[0])

            if frame_num == 31:
                frame_num = 0
                if len(frames_grouped_diff_mobilenet_embeddings) == 10:
                    prediction = two_streams_model.predict(
                        [np.expand_dims(np.asarray(frames_grouped_mobilenet_embeddings), axis=0),
                         np.expand_dims(np.asarray(frames_grouped_diff_mobilenet_embeddings), axis=0)])
                    if prediction > 0.5:
                        # is_Fight = True
                        conn.send([[0, 0, original.shape[1], original.shape[0], 3]])
                    else:
                        # if is_Fight:
                        #     conn.send([[0, 0, original.shape[1], original.shape[0], 3]])
                        # else:
                        conn.send(None)
        conn.send(None)



        cv2.imshow('frame', original)
        if cv2.waitKey(1) == 27:
            break
