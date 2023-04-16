import multiprocessing as mp
from time import sleep

import cv2
from vidgear.gears import CamGear


class Stream:
    def __init__(self, source):
        self.__name = source[0]
        self.__source_url = source[1]
        self.parent_conn, child_conn = mp.Pipe()
        self.p = mp.Process(target=self.update, args=(child_conn,))
        self.p.daemon = True
        self.p.start()

    def update(self, conn):
        cap = CamGear(source=0, logging=True).start()
        # cap = CamGear(source='D:\\Datasets\\FallVideo\\Fall & Daily Activities Video Sample\\Video3.avi', logging=True).start()
        # cap = CamGear(source='D:\Datasets\\Demo2.mp4', logging=True).start()
        run = True
        while run:
            rec_dat = conn.recv()
            if rec_dat == 1:
                frame = cap.read()
                conn.send(frame)
            elif rec_dat == 2:
                cap.stop()
                run = False
        conn.close()

    def disconnect(self):
        self.parent_conn.send(2)

    def get_name(self):
        return self.__name

    def get_source_url(self):
        return self.__source_url

    def get_feed(self):
        self.parent_conn.send(1)
        # sleep(0.08)
        frame = self.parent_conn.recv()
        self.parent_conn.send(0)
        return frame
