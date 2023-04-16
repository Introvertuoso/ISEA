class Template:
    def __init__(self,cnt, cnt_x, cnt_y, cnt_area, bb_x, bb_y, bb_width, bb_height, hog_features,sift_keypoints,score = 0):
        self.cnt = cnt
        self.cnt_x = cnt_x
        self.cnt_y = cnt_y
        self.cnt_area = cnt_area
        self.bb_x = bb_x
        self.bb_y = bb_y
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.hog_features = hog_features
        self.sift_keypoints = sift_keypoints
        self.score = score

    def get_cnt(self):
        return self.cnt

    def get_cnt_x(self):
        return self.cnt_x

    def get_cnt_y(self):
        return self.cnt_y

    def get_cnt_area(self):
        return self.cnt_area

    def get_bb_x(self):
        return self.bb_x

    def get_bb_y(self):
        return self.bb_y

    def get_bb_width(self):
        return self.bb_width

    def get_bb_height(self):
        return self.bb_height

    def get_bb_area(self):
        return self.bb_height * self.bb_width

    def get_hog_features(self):
        return self.hog_features

    def get_sift_keypoints(self):
        return self.sift_keypoints

    def get_score(self):
        return self.score

    def increase_score(self):
        self.score = self.score + 1
