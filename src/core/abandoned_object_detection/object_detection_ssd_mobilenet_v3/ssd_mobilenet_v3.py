import cv2
class DetectionModel:
    def __init__(self, thres=0.45):
        self.thres = thres
        # Threshold to detect object

        self.classNames = []
        classFile = 'C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\abandoned_object_detection' \
                    '\\object_detection_ssd_mobilenet_v3\\coco.names'
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        configPath = 'C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\abandoned_object_detection' \
                     '\\object_detection_ssd_mobilenet_v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\abandoned_object_detection' \
                      '\\object_detection_ssd_mobilenet_v3\\frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def detect(self, img):
        classIds, confs, bbox = self.net.detect(img, confThreshold=self.thres)
        objects = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                objects.append(self.classNames[classId-1])
        return objects