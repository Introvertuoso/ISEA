def init():
    global last_uvicorn_port
    last_uvicorn_port = -1

    global port_controller_map
    port_controller_map = { }