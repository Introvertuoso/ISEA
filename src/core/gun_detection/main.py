import torch
import sys
import platform

sys.path.insert(0, 'core/gun_detection')


# Used for
def start(conn):
    # Model
    path = ''
    if platform.system() == 'Linux':
        path = '/home/nader_adi/Ite/fifth year/graduation/back-end/src/core/gun_detection/weights/last.pt'
    else:
        path = 'C:\\Users\\ASUS\\PycharmProjects\\graduation-back-end\\src\\core\\gun_detection\\weights\\last.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    model.conf = 0.7  # confidence threshold (0-1)
    while True:
        img = conn.recv()

        # Inference
        results = model(img)
        #     print(att, getattr(results, att))
        # results.print()
        temp = results.pandas().xyxy
        count = len(temp[0].xmin)
        if count > 0:
            rec_points = []
            for i in range(count):
                rec_points.append(
                    [int(temp[0].xmin[i]), int(temp[0].ymin[i]), int(temp[0].xmax[i]), int(temp[0].ymax[i]), 1]) # last index is detector ID in DB
            conn.send(rec_points)
        else:
            conn.send(None)
