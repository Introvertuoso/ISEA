import cv2
import numpy as np
from src.core.abandoned_object_detection.candidates_handler import CandidatesHandler
from src.core.abandoned_object_detection.img_processing import mog_mask_processing, pre_process, diff_process, \
    check_illumination_changes, cso_img_process
from src.core.abandoned_object_detection.states import MaskState, CSOState


def start(conn):
    init = True
    rgb = conn.recv()
    kernel_2_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    pso_lock = False

    long = cv2.createBackgroundSubtractorMOG2()
    short = cv2.createBackgroundSubtractorMOG2()

    learning_rate_long = 0.0003
    learning_rate_short = 0.008


    final_csos = np.zeros((rgb.shape[0], rgb.shape[1]), np.uint8)
    candidates_handler = CandidatesHandler()

    frame_counter = 0

    first_pso = True
    pso_frame_num = 0
    pso_counter = 0

    first_cso = False
    cso_frame_num = 0
    cso_counter = 0
    while 1:
        if frame_counter % 2 != 0:
            conn.send(None)
        frame_counter += 1
        boxes = []
        # read frames
        if not init:
            rgb = conn.recv()
        init = False
        img = pre_process(rgb)
        video_frame_processed = cso_img_process(rgb)
        if frame_counter <= 60:
            lb = rgb
        if frame_counter == 60:
            averageValue1 = np.float32(rgb)
        elif frame_counter >= 60:
            cv2.accumulateWeighted(rgb, averageValue1, 0.005)
            lb = cv2.convertScaleAbs(averageValue1)
            cv2.imshow('lb', lb)
            cv2.waitKey(1)
        long_mask = long.apply(img, None, learning_rate_long)
        long_mask = mog_mask_processing(long_mask, kernel=kernel_2_2)

        short_mask = short.apply(img, None, learning_rate_short)
        short_mask = mog_mask_processing(short_mask, kernel=kernel_2_2)

        diff = cv2.bitwise_xor(src1=short_mask, src2=long_mask)
        diff = diff_process(diff)
        cv2.imshow('diff', diff)
        if check_illumination_changes(diff):
            learning_rate_long = 1
            learning_rate_short = 1
        elif not check_illumination_changes(diff):
            learning_rate_long = 0.0003
            learning_rate_short = 0.008

        " --- --- --- PSO to CSO --- --- ---"

        if not pso_lock:
            if first_pso:
                state = candidates_handler.extract_candidates(diff_mask=diff, rgb_frame=rgb, prev_frames=lb,
                                                              initital=True)
                if state == MaskState.NOT_VALID_MASK:
                    conn.send(None)
                else:
                    first_pso = False

            else:
                pso_counter = pso_counter + 1
                if pso_counter == 10:
                    state = candidates_handler.extract_candidates(diff_mask=diff, rgb_frame=rgb)
                    if state == MaskState.VALID_MASK:
                        pso_frame_num = pso_frame_num + 1
                    else:
                        pso_frame_num = 0
                        first_pso = True
                    pso_counter = 0

            if pso_frame_num == 5:
                csos_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                csos = candidates_handler.find_csos()
                if len(csos) != 0:
                    for cso in csos:
                        cv2.drawContours(csos_mask, [cso], 0, 255, -1)
                csos_filtered = candidates_handler.csos_filter(csos_mask, rgb)
                final_csos = csos_filtered
                first_pso = False
                pso_lock = True
                pso_frame_num = 0
                first_cso = True

        " --- --- --- CSO to SO --- --- --- "

        if first_cso:
            cso_counter = cso_counter + 1
            if cso_counter == 20:
                cso_frame_num = cso_frame_num + 1
                video_frame_processed = cso_img_process(rgb)
                state = candidates_handler.cso_to_so(final_csos, rgb, video_frame_processed, cso_frame_num,
                                                     initial=True if cso_frame_num == 1 else False)

                if state == CSOState.BREAK or state == CSOState.CSOS_NOT_VALID:

                    cso_counter = 0
                    cso_frame_num = 0
                    first_cso = False
                    first_pso = True
                    pso_lock = False
                    conn.send(None)

                else:
                    cso_counter = 0
                    if isinstance(state, list):
                        sos = state
                        cso_frame_num = 0

                        first_cso = False
                        first_pso = True
                        pso_lock = False
                        count = 0

                        for sovf, csodf, bb in sos:
                            count = count + 1
                            x, y, w, h = bb
                            boxes.append([x, y, x + w, y + h, 2])
                        conn.send(boxes)
                    else:
                        conn.send(None)
            else:
                conn.send(None)
        else:
            conn.send(None)
