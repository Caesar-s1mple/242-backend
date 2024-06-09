import os
import random
import time
from datetime import datetime, timedelta
import requests
import json

URL = 'http://123.60.2.198:8123/push_data'
type_list = ['境外新闻', '社交媒体', '消息应用', '问答社区']
start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 11, 1)

for file in os.listdir('./database/sample'):
    with open(os.path.join('./database/sample', file), 'r', encoding='utf-8') as f:
        for line in f.readlines()[:200]:
            json_line = json.loads(line)
            content = json_line['text']
            params = {
                'type': '境外新闻',
                'date_time': datetime.strftime(start_date + timedelta(days=random.randint(0, (end_date - start_date).days)), '%Y-%m-%d %H:%M:%S'),
                'content': content
            }
            requests.post(URL, json=params)
            time.sleep(0.1)
