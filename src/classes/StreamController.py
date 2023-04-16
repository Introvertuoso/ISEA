from src.classes.ProcessController import ProcessController


class StreamController:
    def __init__(self):
        self.__streams = []
        self.__process_controllers = []

    def run(self):
        for pc in self.__process_controllers:
            pc.start()

    def add_stream(self, stream, tasks):
        self.__streams.append(stream)
        self.__add_process_controller(stream, tasks)

    def remove_stream(self, value):
        self.__remove_process_controller(value)

    def find_stream(self, name):
        result = None
        for stream in self.__streams:
            if stream.get_name() == name:
                result = stream
        return result

    def __add_process_controller(self, stream, tasks):
        detect_falls, detect_fights, detect_abandoned_objects, detect_guns = None, None, None, None
        for task in tasks:
            if task['id'] == 1:
                detect_guns = task
            elif task['id'] == 2:
                detect_abandoned_objects = task
            elif task['id'] == 3:
                detect_fights = task
            elif task['id'] == 4:
                detect_falls = task
        pc = ProcessController(stream,
                               detect_falls=detect_falls,
                               detect_fights=detect_fights,
                               detect_abandoned_objects=detect_abandoned_objects,
                               detect_guns=detect_guns)
        pc.daemon = True
        self.__process_controllers.append(pc)

    def __remove_process_controller(self, stream):
        print('Terminating Each Process Stream (Disconnect)')
        temp = None
        for pc in self.__process_controllers:
            if pc.get_stream() is stream:
                temp = pc
                pc.terminate()
                stream.disconnect()
                self.__streams.remove(stream)
        self.__process_controllers.remove(temp)

    def terminate(self):
        print('Terminating Stream Controller')
        for i in range(len(self.__streams)):
            self.remove_stream(self.__streams[0])

    @property
    def streams(self):
        return self.__streams

    @property
    def process_controllers(self):
        return self.__process_controllers
