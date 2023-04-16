import datetime
from flask import Flask, _app_ctx_stack, jsonify, request, url_for
# from flask_migrate import Migrate
# from flask_cors import CORS
from sqlalchemy.orm import scoped_session
from flask_cors import CORS, cross_origin
import bcrypt

from main import start_logic_controller, terminate
from . import models
from .database import SessionLocal, engine

import src.globals as globals_vars

GUN_DETECT_COUNTER = 5
FIGHT_DETECT_COUNTER = 1
ABANDON_DETECT_COUNTER = 1
FALL_DETECT_COUNTER = 15

from pyfcm import FCMNotification
# Firebase Cloud Messaging init
push_service = FCMNotification(api_key="REDACTED")

models.Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app)

app.session = scoped_session(SessionLocal, scopefunc=_app_ctx_stack.__ident_func__)

db = SessionLocal()

# migrate = Migrate(app, db)


# Creating Static Role On Server Init
# Creating AI Tasks On Server Init
check_roles = app.session.query(models.Role).all()
if check_roles is None or len(check_roles):
    owner_role = models.Role(
        name='OWNER',
        clearance_level=3,
        creation_date=datetime.datetime.now()
    )

    db.add(owner_role)

# Creating AI Tasks On Server Init
check_tasks = app.session.query(models.Task).all()
if check_tasks is None or len(check_tasks) == 0:
    gun_detection_task = models.Task(
        name="Gun_Detection",
        bb_color="#00ff00",
        creation_date=datetime.datetime.now()
    )

    abandoned_object_task = models.Task(
        name="Abandoned_Object",
        bb_color="#ff0000",
        creation_date=datetime.datetime.now()
    )

    fight_detector_task = models.Task(
        name="Fight_Detector",
        bb_color="#0000ff",
        creation_date=datetime.datetime.now()
    )

    fall_detector_task = models.Task(
        name="Fall_Detector",
        bb_color="#ffff00",
        creation_date=datetime.datetime.now()
    )
    db.add_all([gun_detection_task, abandoned_object_task, fight_detector_task, fall_detector_task])
db.commit()
db.close()


globals_vars.init()

@app.route("/")
def hello_world():
    return "<p style='font-size: 100px;'>Dunder Mifflen, This is Pam!</p>"


@app.route("/api/companies", methods=["GET", "POST"])
def companies():
    if request.method == "GET":
        companies = app.session.query(models.Company).all()
        result = [company.to_dict() for company in companies]
        return jsonify({"result": result, "code": 200})
    elif request.method == "POST":
        # Request content-type must be application/json...
        req_company = request.get_json(silent=True)
        print(req_company['name'])
        db_company = models.Company(
            name=req_company['name'],
            phone_number=req_company['phone_number'],
            address=req_company['address'],
            email=req_company['email'],
            is_active=True,
            creation_date=datetime.datetime.now(),
            logo=req_company['logo']
        )
        db.add(db_company)
        db.commit()
        response = jsonify({"result": db_company.to_dict(), "code": 200, })
        db.close()
        return response


@app.route("/api/users", methods=["GET", "POST"])
def users():
    if request.method == "GET":
        users = app.session.query(models.User).all()
        result = [user.to_dict() for user in users]
        return jsonify({"result": result, "code": 200})
    elif request.method == "POST":
        req_user = request.get_json(silent=True)
        req_password = req_user['password']
        hashed_password = hash_user_password(req_password)
        db_user = models.User(
            full_name=req_user['fullname'],
            username=req_user['username'],
            phone_number=req_user['phone_number'],
            email=req_user['email'],
            role_id=req_user['role_id'],
            company_id=req_user['company_id'],
            is_active=True,
            creation_date=datetime.datetime.now(),
        )
        db_password = models.Password(
            password=hashed_password,
            creation_date=datetime.datetime.now(),
            user_id=db_user.id
        )

        db.add(db_user)
        db.add(db_password)
        db.commit()
        response = jsonify({"result": db_user.to_dict(), "code": 200, })
        db.close()
        return response


def hash_user_password(plain_password):
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode("utf-8")


@app.route("/api/users/<int:id>", methods=["GET"])
def users_by_id(id):
    if request.method == "GET":
        user = app.session.query(models.User).filter_by(id=id).first()
        return jsonify({"result": user.to_dict(), "code": 200, })


@app.route("/api/cameras", methods=["GET", "POST"])
def cameras():
    if request.method == "GET":
        cameras = app.session.query(models.Camera).all()
        result = [camera.to_dict() for camera in cameras]
        return jsonify({"result": result, "code": 200})

    elif request.method == "POST":
        req_camera = request.get_json(silent=True)
        company = app.session.query(models.Company).filter_by(id=req_camera['company_id']).first()
        if company is None:
            return jsonify({"result": "Bad Request, Company ID Not Found", "code": 400})

        company = company.to_dict()

        db_camera = models.Camera(
            name=req_camera['name'],
            feed_url=req_camera['feed_url'],
            fps=req_camera['fps'],
            resolution=req_camera['resolution'],
            creation_date=datetime.datetime.now(),
            company_id=company['id']
        )
        db.add(db_camera)
        db.commit()
        response = jsonify({"result": db_camera.to_dict(), "code": 200, })
        db.close()
        return response


@app.route("/api/tasks", methods=["GET"])
def tasks():
    tasks = app.session.query(models.Task).all()
    result = [task.to_dict() for task in tasks]
    return jsonify({"result": result, "code": 200})


@app.route("/api/tasks/start", methods=["POST"])
@cross_origin()
def start_task():
    # request object should contain camera_id, tasks IDs [int, int, int, ...]
    if request.method == "POST":
        req_config = request.get_json(silent=True)
        if req_config['tasks'] is None or len(req_config['tasks']) == 0:
            return jsonify({"result": "Bad Request, Must select at least one task to run", "code": 400})
        camera = app.session.query(models.Camera).filter_by(id=int(req_config['camera_id']),
                                                            company_id=int(req_config['company_id'])).first()
        if camera is None:
            return jsonify({"result": "Bad Request, Camera not found.", "code": 400})

        # Check if process already running
        for t_id in req_config['tasks']:
            running_tasks = app.session.query(models.tasks_cameras) \
                .filter_by(camera_id=req_config['camera_id'], task_id=t_id, is_active=True).all()
            print(running_tasks)
            if len(running_tasks) > 0:
                return jsonify({"result": "Service already running", "code": 422})
        # Starting the required services...
        tasks = []
        # Use this instead of loop ...
        # ran_tasks = app.session.query(models.Task).filter(models.Task.id.in_(req_config['tasks']))
        for t_id in req_config['tasks']:
            db_task = app.session.query(models.Task).filter_by(id=t_id).one()
            tasks.append(db_task.to_dict())
        print(tasks)
        controller = start_logic_controller(camera, tasks)
        # All Good AI tasks created successfully...
        for t in tasks:
            db_tasks_camera_statement = models.tasks_cameras.insert() \
                .values(camera_id=req_config['camera_id'],
                        task_id=t['id'],
                        stream_port=globals_vars.last_uvicorn_port,
                        creation_date=datetime.datetime.now(),
                        is_active=True)
            db_execute = app.session.execute(db_tasks_camera_statement)
            # print(db_execute.all()) # This should work, but connection is closed for some reasons...
            app.session.commit()
            app.session.close()

            # Update port_controller map
            globals_vars.port_controller_map[globals_vars.last_uvicorn_port] = controller
            # Clients should call /api/tasks/active/camera/<int:id> to get active tasks list...
        return jsonify({"result": {
            "stream_url": 'http://localhost:' + str(globals_vars.last_uvicorn_port) + '/video',
            "tasks":[t for t in tasks]
        },
            "code": 200})



@app.route("/api/tasks/terminate/<int:port>", methods=["GET"])
def termiante_selected_task(port):
    controller = globals_vars.port_controller_map[port]
    controller.terminate()
    # Update Camera_Task relation Active status
    running_tasks = app.session.query(models.tasks_cameras) \
        .filter_by(stream_port=port, is_active=True).update(dict(is_active=False))
    app.session.commit()
    app.session.close()
    return jsonify({"result": "Tasks Terminated Successfully", "code": 200})


@app.route("/api/tasks/reset_all", methods=["GET"])
def reset_all_running_tasks():
    # For development usage ONLY
    if request.method == "GET":
        num_rows_deleted = app.session.query(models.tasks_cameras).delete()
        app.session.commit()
        app.session.close()
        return jsonify({"num_rows_deleted": num_rows_deleted})

@app.route("/api/tasks/active/camera/<int:id>", methods=["GET"])
def get_tasks_active(id):
    if request.method == "GET":
        running_tasks = app.session.query(models.tasks_cameras) \
            .filter_by(camera_id=id, is_active=True).all()
        print(running_tasks)
        if len(running_tasks) == 0:
            return jsonify({"result":"No active tasks on the selected camera", "code": 422})

        tasks = []
        for t in running_tasks:
            # t = t.to_dict()
            db_task = app.session.query(models.Task).filter_by(id=t['task_id']).one()
            tasks.append(db_task.to_dict())
        return jsonify({"result" : {
            "tasks": tasks,
            "stream_url" : 'http://localhost:' + running_tasks[0]['stream_port'] + '/video'  # [0] is definitely available
        },  "code" : 200})


def insert_to_alerts(port_number, task_id, frame, task_counter):
    camera_task = app.session.query(models.tasks_cameras) \
        .filter_by(is_active=True, task_id=task_id, stream_port=port_number).first()
    # camera_task = camera_task.to_dict()

    db_alert = models.Alert(
        camera_task_id=camera_task['id'],
        frame=str(frame)
    )
    db.add(db_alert)

    # if task_id == 1:
    #     # GUN
    #     if task_counter == GUN_DETECT_COUNTER:

    db.commit()
    db.close()


@app.route("/api/notifications", methods=["POST"])
def send_notification():
    if request.method == "POST":
        registration_id = ""
        message_title = "Uber update"
        message_body = "Hi john, your customized news for today is ready"
        result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title,
                                                   message_body=message_body);
        print(result)
        return jsonify({"code" : 200})
