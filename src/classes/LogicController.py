from src.classes.Stream import Stream
from src.classes.StreamController import StreamController


class LogicController:

    def __init__(self, cam, tasks):
        self.__source = cam.feed_url
        self.__tasks = tasks

        self.__stream_controller = None

    def is_valid(self):
        return self.__source is not None

    def run(self):
        """
        -populate streams list (add stream) ---> Populated on construction now ...

        -for each stream create Process C thread, update list of tuples
        -start all create process C thread, update OS ports pool
        """

        self.__stream_controller = StreamController()

        if self.is_valid():
            stream = Stream(self.__source)
            self.__stream_controller.add_stream(stream, self.__tasks)

        self.__stream_controller.run()
        return self.__stream_controller

    def terminate(self):
        print('Terminating Logic Controller')
        print(self.__stream_controller)
        self.__stream_controller.terminate()
        self.__stream_controller = None
        self.__source = ''
        self.__tasks = []

    @property
    def source(self):
        return self.__source

    @property
    def stream_controller(self):
        return self.__stream_controller
