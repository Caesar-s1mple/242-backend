from sqlalchemy import Column, Integer, Boolean, Text, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DailyData(Base):
    __tablename__ = 'daily_data'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    sensitive = Column(Boolean)
    sensitive_score = Column(Float)
    is_bot = Column(Boolean)
    is_bot_score = Column(Float)
    model_judgment = Column(Text)
    url = Column(Text)
    topic = Column(Text)


class Data(Base):
    __tablename__ = 'data'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    sensitive = Column(Boolean)
    sensitive_score = Column(Float)
    is_bot = Column(Boolean)
    is_bot_score = Column(Float)
    model_judgment = Column(Text)
    url = Column(Text)
    topic = Column(Text)
    uuid = Column(Text)


class HumanLLM(Base):
    __tablename__ = 'human_llm'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    is_bot = Column(Boolean)
    is_bot_score = Column(Float)
    model_judgment = Column(Text)
    url = Column(Text)
    topic = Column(Text)
    auditor = Column(Text)
    uuid = Column(Text)
    merged = Column(Boolean)


class Sen(Base):
    __tablename__ = 'sen'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    sensitive = Column(Boolean)
    sensitive_score = Column(Float)
    url = Column(Text)
    topic = Column(Text)
    auditor = Column(Text)
    uuid = Column(Text)
    merged = Column(Boolean)


class TempHumanLLM(Base):
    __tablename__ = 'temp_human_llm'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    is_bot = Column(Boolean)
    is_bot_score = Column(Float)
    model_judgment = Column(Text)
    url = Column(Text)
    topic = Column(Text)
    username = Column(Text)
    uuid = Column(Text)


class TempSen(Base):
    __tablename__ = 'temp_sen'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text)
    time = Column(Text)
    push_time = Column(Text)
    content = Column(Text)
    sensitive = Column(Boolean)
    sensitive_score = Column(Float)
    url = Column(Text)
    topic = Column(Text)
    username = Column(Text)
    uuid = Column(Text)


class Users(Base):
    __tablename__ = 'users'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(Text, unique=True)
    password = Column(Text)
    role = Column(Text)  # C_admin: 国家中心管理员, C_user: 国家中心普通用户, user: 分中心用户


class LLMRank(Base):
    __tablename__ = 'llm_rank'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    second_level = Column(Text)
    first_level = Column(Text)
    metric = Column(Text)
    model = Column(Text)
    score = Column(Float)
