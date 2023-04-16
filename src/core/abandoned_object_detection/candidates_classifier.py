import cv2
import numpy as np

from src.core.abandoned_object_detection.states import CSOState
from src.core.abandoned_object_detection.states import MaskState


class CandidatesClassifier:

    def make_decision(self, video_frame, video_frame_processed, csodf, long_background, long_background_processed,
                      detect):
        "Perfrom object segmentation on the VF (video_frame) using Mask R-CNN (SOvf)"
        sovf = self.get_mask(video_frame, video_frame_processed, detect)
        match_thresh = 75
        if not isinstance(sovf, MaskState):
            matching_result = self.match(csodf, sovf[0])
            if matching_result > match_thresh:
                return CSOState.ABANDONED_OBJECT
            else:
                if 'person' in detect(long_background):
                    return CSOState.GHOST_REGION
                "Perform object segmentation on the LB (long_background) using Mask R-CNN (SOlb)"
                solb = self.get_mask(long_background, long_background_processed, detect)
                if isinstance(solb, MaskState):
                    return CSOState.ABANDONED_OBJECT
                else:
                    matching_result = self.match(sovf[0], solb[0])
                    if matching_result > match_thresh:
                        return CSOState.GHOST_REGION
                    elif self.area(solb[1]) < self.area(sovf[1]):
                        return CSOState.ABANDONED_OBJECT
                    elif self.area(solb[1]) > self.area(sovf[1]):
                        return CSOState.STOLEN_OBJECT
        else:
            if 'person' in detect(long_background):
                return CSOState.GHOST_REGION
            "Perform object segmentation on the LB (long_background) using Mask R-CNN (SOlb)"
            solb = self.get_mask(long_background, long_background_processed, detect)
            if not isinstance(solb, MaskState):
                return CSOState.STOLEN_OBJECT
            else:
                return CSOState.GHOST_REGION

    def area(self, segmented_object1):
        return cv2.contourArea(segmented_object1)

    def match(self, segmented_object1, segmented_object2):
        n = np.count_nonzero(segmented_object1 == 255)
        m = np.count_nonzero(np.array([1 for i in range(segmented_object1.shape[0])
                                       for j in range(segmented_object1.shape[1])
                                       if segmented_object1[i][j] == 255 and segmented_object2[i][j] == 255]) == 1)
        return (m / n) * 100

    def get_mask(self, frame, frame_processed, detect):
        objects = detect(frame)
        if len(objects) == 0:
            return MaskState.NOT_VALID_MASK
        else:
            contours, _ = cv2.findContours(frame_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return MaskState.NOT_VALID_MASK
            cnts_sorted = sorted(contours, reverse=True, key=lambda cnt: cv2.contourArea(cnt))
            max_area = cnts_sorted[0]
            mask = np.zeros(frame_processed.shape, np.uint8)
            cv2.drawContours(mask, [max_area], 0, 255, -1)
            return mask, max_area
