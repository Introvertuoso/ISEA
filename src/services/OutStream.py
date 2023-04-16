import socket
from threading import Thread

from vidgear.gears import NetGear
import uvicorn, asyncio, cv2
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer


class StreamAiOutput(Thread):

    def __init__(self, communication_port, streaming_port):
        self.__communication_port = communication_port
        self.__streaming_port = streaming_port
        self.__started = True
        super().__init__()

    # create your own custom frame producer
    async def streaming_frame_producer(self):

        # loop over frames
        while self.__started:
            # read frame from provided source
            frame = self.__client.recv()

            # reducer frames size if you want more performance otherwise comment this line
            frame = await reducer(frame, percentage=30)  # reduce frame by 30%
            # handle JPEG encoding
            encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
            # yield frame in byte format
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n"
            await asyncio.sleep(0.0000001)

    def run(self):
        print('start_streaming_client')
        try:
            self.__client = NetGear(
                # TODO: Make this dynamic too
                address=str(socket.gethostbyname(socket.gethostname())),
                port=self.__communication_port,
                protocol="tcp",
                receive_mode=True,
                logging=True,
            )
            self.__web = WebGear(logging=True)

        except Exception:
            print('Stream AI Output Exception')
            print(Exception)

        # finally:
        #     self.__client.close()
        # add your custom frame producer to config

        try:
            self.__web.config["generator"] = self.streaming_frame_producer

            # run this app on Uvicorn server at address http://localhost:8000/
            uvicorn.run(self.__web(), host="localhost", port=self.__streaming_port)

        except:
            print('Exception encountered when trying to start streaming server')

        finally:
            pass
            # close app safely
            # self.__web.shutdown()

    def terminate(self):
        self.__started = False
        self.__client.close()
        self.__web.shutdown()
