import cv2
import numpy as np

from src.core.abandoned_object_detection.candidates_classifier import CandidatesClassifier
from src.core.abandoned_object_detection.object_detection_ssd_mobilenet_v3.ssd_mobilenet_v3 import DetectionModel
from src.core.abandoned_object_detection.img_processing import cso_img_process, draw_largest_contour

np.seterr(divide='ignore', invalid='ignore')
from src.core.abandoned_object_detection.states import MaskState, CSOState
from src.core.abandoned_object_detection.template import Template
from src.core.abandoned_object_detection.img_processing import enlargeBB

luggage = ['suitcase', 'backpack', 'handbag']


class CandidatesHandler:

    def __init__(self):
        self.initial_templates = []
        self.so_scores = []
        self.model = DetectionModel()
        self.classifier = CandidatesClassifier()
        self.first_rgb = None
        self.prev_frames = None

    def extract_candidates(self, diff_mask, rgb_frame, prev_frames=None, initital=False):
        if not initital:
            current_candidatets = []
        else:
            self.initial_templates = []
            self.first_rgb = rgb_frame
            self.prev_frames = prev_frames
        contours, hierarchy = cv2.findContours(diff_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return MaskState.NOT_VALID_MASK
        else:
            for cnt in contours:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cnt_area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                bx = x + w / 2
                by = y + h / 2
                hog_features = self.calc_hog(rgb_frame, x, y, w, h)
                sift_features = self.calc_sift(rgb_frame, x, y, w, h)
                if initital:
                    template = Template(cnt, cx, cy, cnt_area, bx, by, w, h, hog_features, sift_features)
                    self.initial_templates.append(template)
                else:
                    template = Template(cnt, cx, cy, cnt_area, bx, by, w, h, hog_features, sift_features)
                    current_candidatets.append(template)
            if not initital:
                matching_state = self.compare_candidates(current_candidatets)
                if matching_state == MaskState.NO_MATCHES:
                    return MaskState.NOT_VALID_MASK
                else:
                    return MaskState.VALID_MASK
            else:
                return MaskState.FIRST_VALID_MASK

    def compare_candidates(self, current_candidates):
        matches = 0
        for curr_cnt in current_candidates:
            for init_cnt in self.initial_templates:

                init_cnt_bb_x = init_cnt.get_bb_x()
                init_cnt_bb_y = init_cnt.get_bb_y()

                curr_cnt_bb_x = curr_cnt.get_bb_x()
                curr_cnt_bb_y = curr_cnt.get_bb_y()

                rad = min(curr_cnt.get_bb_width(), curr_cnt.get_bb_height()) / 4

                if self.is_close(curr_cnt_bb_x, curr_cnt_bb_y, rad, init_cnt_bb_x, init_cnt_bb_y):
                    matching_score = 0
                    if self.is_similar_sift(curr_cnt.get_sift_keypoints(), init_cnt.get_sift_keypoints()):
                        matching_score = matching_score + 1
                    if self.is_similar_area(curr_cnt.get_cnt_area(), init_cnt.get_cnt_area()):
                        matching_score = matching_score + 1
                    if matching_score == 2:
                        init_cnt.increase_score()
                        matches = matches + 1
                        break
        if matches > 0:
            return MaskState.MATCH_EXIST
        else:
            return MaskState.NO_MATCHES

    def is_similar_sift(self, sift1, sift2):
        bf = cv2.BFMatcher()
        if sift1[1] is None or sift2[1] is None:
            return False
        matches = bf.knnMatch(sift1[1], sift2[1], k=2)
        if len(matches) == 0:
            return False
        good = []
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            except ValueError:
                return False
        percentage = (len(good) * 100) / len(sift1[0])
        if percentage >= 10:
            return True
        else:
            return False

    def is_similar_hog(self, hog1, hog2):
        thresh = 0.5
        hog1 = hog1 / np.linalg.norm(hog1)
        hog2 = hog2 / np.linalg.norm(hog2)
        distance = cv2.norm(hog1, hog2, cv2.NORM_L2)
        if distance < thresh:
            return True
        return False

    def is_similar_area(self, cnt_area1, cnt_area2):
        thresh = 0.75
        similarity = min(cnt_area1, cnt_area2) / max(cnt_area1, cnt_area2)
        if similarity >= thresh:
            return True
        return False

    def is_similar_bb(self, bb_area1, bb_area2):
        thresh = 0.75
        similarity = min(bb_area1, bb_area2) / max(bb_area1, bb_area2)
        if similarity >= thresh:
            return True
        return False

    def is_close(self, circle_x, circle_y, rad, x, y):
        if (x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad:
            return True
        else:
            return False

    def calc_sift(self, rgb_frame, x, y, w, h, crop=True):
        if crop:
            candidate = rgb_frame[y:y + h, x:x + w]
        else:
            candidate = rgb_frame
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(candidate, None)
        return (kp1, des1)

    def calc_hog(self, rgb_frame, x, y, w, h, crop=True):
        if crop:
            candidate = rgb_frame[y:y + h, x:x + w]
        else:
            candidate = rgb_frame
        candidate = cv2.resize(candidate, (32, 32))
        blockSize = (8, 8)
        blockStride = (8, 8)
        cellSize = (4, 4)
        nbins = 9
        winSize = (candidate.shape[1], candidate.shape[0])
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        locations = []
        hist = hog.compute(candidate, None, None, locations)
        hist = np.array(hist)
        mat = np.float32(hist)
        return mat

    def find_csos(self):
        csos = []
        for template in self.initial_templates:
            if template.get_score() == 5:
                csos.append(template.get_cnt())
        return csos

    def csos_filter(self, csos_mask, img):
        filtered = np.zeros(csos_mask.shape, np.uint8)
        contours, hierarchy = cv2.findContours(csos_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x, y, w, h = enlargeBB(csos_mask.shape[0], csos_mask.shape[1], x, y, w, h, padding=20)
                box = img[y:y + h, x:x + w]
                objects = self.model.detect(box)
                if len(objects) >= 1 and 'person' in objects:
                    continue
                else:
                    cv2.drawContours(filtered, [cnt], 0, 255, -1)
        return filtered

    def cso_to_so(self, csos_mask, video_frame_rgb,video_frame_processed, counter, initial):
        if initial:
            self.so_scores = []
        contours, _ = cv2.findContours(csos_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return CSOState.CSOS_NOT_VALID
        elif len(contours) == 1:
            if initial:
                self.so_scores.append(0)
            self.compare_first_cso_with_current_vf(contours[0], 0, csos_mask, video_frame_rgb)
        elif len(contours) > 1:
            if initial:
                for i in range(len(contours)):
                    self.so_scores.append(0)
            for i in range(len(contours)):
                self.compare_first_cso_with_current_vf(contours[i], i, csos_mask, video_frame_rgb)
        if counter == 3:
            arr = np.array(self.so_scores)
            if np.isin(arr, [0]).all():
                return CSOState.BREAK
        if counter == 5:
            sos = []
            for score, cso in zip(self.so_scores, contours):
                if score >= 3:
                    x, y, w, h = cv2.boundingRect(cso)
                    x, y, w, h = enlargeBB(csos_mask.shape[0], csos_mask.shape[1], x, y, w, h, padding=10)
                    sovf = video_frame_rgb[y:y + h, x:x + w]
                    sovf_precessed = video_frame_processed[y:y + h, x:x + w]
                    csodf = self.first_rgb[y:y + h, x:x + w]
                    csodf_processed = csos_mask[y:y + h, x:x + w]
                    solb = self.prev_frames[y:y + h, x:x + w]
                    solb_processed = cso_img_process(solb)
                    solb_processed = draw_largest_contour(solb_processed)
                    sos.append((sovf, csodf, (x, y, w, h)))
                    if len(sos) > 0:
                        decision = self.classifier.make_decision(sovf, sovf_precessed, csodf_processed, solb,
                                                             solb_processed, self.model.detect)
                        if decision != CSOState.ABANDONED_OBJECT:
                            return CSOState.BREAK
                    else:
                        return CSOState.CSOS_VALID
                    print(decision)
            return sos
        return CSOState.CSOS_VALID

    def compare_first_cso_with_current_vf(self, cnt, i, csos_mask, video_frame_rgb):
        score = 0
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = enlargeBB(csos_mask.shape[0], csos_mask.shape[1], x, y, w, h, padding=10)
        first_rgb = self.first_rgb[y:y + h, x:x + w]
        video_frame_rgb = video_frame_rgb[y:y + h, x:x + w]
        video_frame_rgb_sift_features = self.calc_sift(video_frame_rgb, x, y, w, h, crop=False)
        first_rgb_sift_features = self.calc_sift(first_rgb, x, y, w, h, crop=False)
        if self.is_similar_sift(first_rgb_sift_features, video_frame_rgb_sift_features):
            score = score + 1
        if score == 1:
            self.so_scores[i] = self.so_scores[i] + 1
        return
