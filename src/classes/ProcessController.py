import base64
import math
import socket
import time
from multiprocessing import Pipe
from multiprocessing import Process
from threading import Thread

import cv2
import numpy as np

from vidgear.gears import NetGear
from src.Utility import dispatch_port

from src.api_flask import server

import src.core.fall_detection.openpose.build_windows.examples.user_code.fall_detection as fall_detector
import src.core.fight_detection.main as fight_detector
import src.core.abandoned_object_detection.main as abandoned_object_detector
import src.core.gun_detection.main as gun_detector

#  detectors must have a 'start' function or else no workie
# from src.services.OutStream import StreamAiOutput
from src.services.OutStream import StreamAiOutput

import src.globals as globals_vars
# TODO : Add detect_falls, detect_fights, back here with the imports...
detectors = [fall_detector,
             fight_detector,
             abandoned_object_detector,
             gun_detector]


#  make this come from get_name or something in the respective mains
# detector_names = ['fall_detector',
#                   'fight_detector',
#                   'abandoned_object_detector',
#                   'gun_detector']
# detector_colors = [(0, 255, 0),
#                    (255, 0, 0),
#                    (0, 0, 255),
#                    (255, 255, 0)]


class ProcessController(Thread):
    def __init__(self,
                 stream,
                 detect_falls=None,
                 detect_fights=None,
                 detect_abandoned_objects=None,
                 detect_guns=None):

        super().__init__()
        self.__started = True
        # self.__streamAiOutput = StreamAiOutput()
        # TODO : Add detect_falls, detect_fights, back here with the imports...
        task_list = [detect_falls, detect_fights, detect_abandoned_objects, detect_guns]
        print("tasks list", task_list)

        # TODO: THOSE ARE #1
        self.__communication_port = dispatch_port()
        self.__streaming_port = dispatch_port()
        globals_vars.last_uvicorn_port = self.__streaming_port
        self.__streamAiOutput = StreamAiOutput(self.__communication_port, self.__streaming_port)

        print('Communication port ', self.__communication_port)

        # task_list = [ detect_guns]
        self.__stream = stream
        self.__mask = [False if i is None else True for i in task_list]
        self.__detector_colors = [None if i is None else self.hex_to_rgb(i['bb_color']) for i in task_list]
        self.__detector_names = ['' if i is None else i['name'] for i in task_list]
        self.__frame = None
        self.__processes = []
        self.__pipes = []
        self.__threat_counter = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

        detectors_to_use = np.array(detectors)[self.__mask]
        print('Detectors to use', detectors_to_use)
        for detector in detectors_to_use:
            pipe = Pipe()
            self.__pipes.append(pipe)
            p = Process(target=detector.start, args=(pipe[1],))
            p.daemon = True
            self.__processes.append(p)
            p.start()

    def run(self):

        """
            -start AI tasks in parallel (sys.exec, packages, ...)
            -populate process[] list
            -search for process communication
        """
        detector_names_to_use = np.array(self.__detector_names)[self.__mask]
        detector_colors_to_use = np.array(self.__detector_colors)[self.__mask]
        # TODO: THOSE ARE #2 ???????
        self.__streamAiOutput.start()

        print('Starting  Broadcast...')
        broadcasting_opt = {
            "max_retries": 10
        }
        self.__broadcast_ai_output = NetGear(
            # TODO: Make this dynamic
            address=str(socket.gethostbyname(socket.gethostname())),
            port=self.__communication_port,
            protocol="tcp",
            logging=False,
            **broadcasting_opt
        )

        prev_frame_time = 0
        while self.__started:
            self.__frame = self.__stream.get_feed()
            output_ai_frame = self.__frame

            new_frame_time = time.time()

            # cv2.imshow('original', self.__buffer)
            # if cv2.waitKey(1) == 27:
            #     break  # esc to quit

            for pipe, detector_name, detector_color in zip(self.__pipes, detector_names_to_use, detector_colors_to_use):
                pipe[0].send(self.__frame.copy())
                # self.__streamAiOutput.update_current_frame(self.__frame.copy())
                reply = pipe[0].recv()
                if reply is not None:
                    print(reply)
                    detector_color = tuple(int(c) for c in detector_color)
                    for i in reply:
                        x = i[0]
                        y = i[1]
                        w = i[2]
                        h = i[3]
                        cv2.rectangle(output_ai_frame, (x, y), (w, h), detector_color, 5)
                        # Check detector ID
                        id = i[4]
                        # print(id)
                        print('INSERT TO ALERTS')

                        # Transfer output frame to base64
                        retval, buffer_img = cv2.imencode('.jpg', output_ai_frame)
                        base64_output = base64.b64encode(buffer_img)

                        server.insert_to_alerts(self.__streaming_port, id, base64_output, 0)


            """ 
                Stream AI output_ai_frame frames locally to RTC local service
            """
            if prev_frame_time > 0:
                fps = (new_frame_time - prev_frame_time)
                # cv2.putText(output_ai_frame, str(math.floor(fps)),
                #             (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                # print(fps)
            prev_frame_time = new_frame_time

            if output_ai_frame is not None:
                # TODO: THIS IS #3
                self.__broadcast_ai_output.send(output_ai_frame)
                # cv2.imshow('Output', output_ai_frame)
                # cv2.waitKey(1)

    def terminate(self):
        self.__streamAiOutput.terminate()
        self.__broadcast_ai_output.close()
        self.__started = False
        for pipe in self.__pipes:
            pipe[0].close()
        for process in self.__processes:
            process.terminate()


    def get_frame(self):
        return self.__frame

    def get_stream(self):
        return self.__stream

    def set_stream(self, stream):
        self.__stream = stream

    def hex_to_rgb(self, hex):
        h = str(hex).lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))