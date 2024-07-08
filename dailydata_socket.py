import socket
import json
import time
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from utils.table_models import DailyData, PushedDailyData
import threading

socket_host = '127.0.0.1'
send_socket_port = 8124
listen_socket_port = 8125
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.bind((socket_host, listen_socket_port))
listen_socket.listen(5)

DATABASE_DAILY_DATA = 'sqlite:///./database/daily_data.db'
engine_daily_data = create_engine(DATABASE_DAILY_DATA)
SessionLocalDailyData = sessionmaker(bind=engine_daily_data, autoflush=False, autocommit=False)


def send_func():
    daily_data_db = SessionLocalDailyData()

    while True:
        time.sleep(1)
        data_ids = daily_data_db.query(PushedDailyData.data_id).limit(10).all()
        for data_id in data_ids:
            id = data_id.data_id

            daily_data = daily_data_db.query(DailyData).filter(DailyData.ID == id, DailyData.submitted == False).first()

            if not daily_data or daily_data.submitted:
                daily_data_db.execute(delete(PushedDailyData).filter(PushedDailyData.data_id == id))
                daily_data_db.commit()
                continue

            data = [id, daily_data.content]

            try:
                inference_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                inference_socket.connect((socket_host, send_socket_port))
                inference_socket.sendall(json.dumps(data).encode(encoding='utf-8'))
                inference_socket.close()
                daily_data_db.execute(delete(PushedDailyData).filter(PushedDailyData.data_id == id))
                daily_data_db.commit()
            except:
                print('Failed sending')

    # daily_data_db.close()


def listen_func():
    daily_data_db = SessionLocalDailyData()
    print('Listening...')
    while True:
        conn, address = listen_socket.accept()
        print(f"Connection from {address} has been established.")
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        conn.close()
        if len(data):
            results = json.loads(data.decode(encoding='utf-8'))
            if results:
                # print(results)
                daily_data = daily_data_db.query(DailyData).filter(DailyData.ID == results[0]).first()
                daily_data.is_sensitive = results[1]
                daily_data.sensitive_score = results[2]
                daily_data.is_bot = results[3]
                daily_data.is_bot_score = results[4]
                if len(results) == 6:
                    daily_data.model_judgment = results[5]


if __name__ == '__main__':
    listen_thread = threading.Thread(target=listen_func)
    send_thread = threading.Thread(target=send_func)
    listen_thread.start()
    send_thread.start()
