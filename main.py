"""
    The main goal of this function is to start the surveillance process using out structure
    Generating instances as following
    -Logic Controller (Company)
        -has a Stream Controller
            -has list of process Controllers and Streams (for now one static for each)

        then we should call out some other .py file to simulate some AI ...
"""

from src.classes.LogicController import LogicController

global controller

def start_logic_controller(cam, tasks):
    global controller
    controller = LogicController(cam, tasks)
    # company = LogicController(cam, [{"id": 1, "bb_color": "#00ff00", "name": "Gun_detection"}])
    controller.run()
    return controller

def terminate():
    print('company ctrl', controller)
    controller.terminate()


if __name__ == '__main__':
    start_logic_controller({"name": "Hall", "feed_url": "192.168.1.1"}, None)
