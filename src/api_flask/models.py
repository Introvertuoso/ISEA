from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship

from .database import Base
from sqlalchemy_serializer import SerializerMixin


class Company(Base, SerializerMixin):
    __tablename__ = "company"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone_number = Column(String)
    email = Column(String)
    address = Column(Text, )
    logo = Column(Text)  # As base64
    is_active = Column(Boolean, )
    user = relationship("User")
    creation_date = Column(DateTime)
    update_date = Column(DateTime)


class Role(Base, SerializerMixin):
    __tablename__ = "role"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    clearance_level = Column(Integer)
    # user = relationship("User", back_populates="role")
    creation_date = Column(DateTime)
    update_date = Column(DateTime)


class User(Base, SerializerMixin):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    username = Column(String, index=True)
    email = Column(String)
    phone_number = Column(String)
    password = relationship("Password", back_populates="user", uselist=False)
    role_id = Column(Integer, ForeignKey('role.id'))
    # role = relationship("Role", back_populates="user")
    company_id = Column(Integer, ForeignKey('company.id'))
    creation_date = Column(DateTime)
    update_date = Column(DateTime)
    is_active = Column(Boolean)


class Password(Base, SerializerMixin):
    __tablename__ = "password"

    id = Column(Integer, primary_key=True, index=True)
    password = Column(String)
    user_id = Column(Integer, ForeignKey('user.id'))
    user = relationship("User", back_populates="password")
    creation_date = Column(DateTime)
    update_date = Column(DateTime)


class Camera(Base, SerializerMixin):
    __tablename__ = "camera"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    feed_url = Column(String)
    fps = Column(String)
    resolution = Column(String)
    creation_date = Column(DateTime)
    update_date = Column(DateTime)
    company_id = Column(Integer, ForeignKey('company.id'))


class Task(Base, SerializerMixin):
    __tablename__ = "task"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, )
    bb_color = Column(String)
    creation_date = Column(DateTime)
    update_date = Column(DateTime)


tasks_cameras = Table('tasks_cameras', Base.metadata,
                      Column('id', Integer, primary_key=True, index=True),
                      Column('creation_date', DateTime),
                      Column('is_active', Boolean),
                      Column('stream_port', String),
                      Column('camera_id', ForeignKey('camera.id')),
                      Column('task_id', ForeignKey('task.id')),
                      )


class Alert(Base, SerializerMixin):
    __tablename__ = 'alert'
    id = Column(Integer, primary_key=True, index=True)
    camera_task_id = Column(Integer, ForeignKey('tasks_cameras.id'))
    frame = Column(String)
    creation_date = Column(DateTime)
    update_date = Column(DateTime)
