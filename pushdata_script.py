import os
import random
import time
from datetime import datetime, timedelta
import requests
import json

URL = 'http://127.0.0.1:8123/push_data'
type_list = ['境外新闻', '社交媒体', '消息应用', '问答社区']
start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 11, 1)

for file in os.listdir('./database/sample'):
    n = 0
    with open(os.path.join('./database/sample', file), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            url = None
            if file == 'quora.jsonl':
                type = '问答社区'
                content = json_line['text']
                url = json_line['topic_url']
            elif file == 'telegram.jsonl':
                type = '消息应用'
                content = json_line['message']
            elif file == 'twitter.jsonl' or file == 'facebook.jsonl':
                type = '社交媒体'
                content = json_line['post_text']
                url = json_line['id']
            else:
                type = '境外新闻'
                content = json_line['text']

            if not content:
                continue
            params = {
                'type': type,
                'date_time': datetime.strftime(start_date + timedelta(days=random.randint(0, (end_date - start_date).days)), '%Y-%m-%d %H:%M:%S'),
                'content': content,
                'url': url
            }
            print(params)
            requests.post(URL, json=params)
            # time.sleep(0.1)
            n += 1
            if n == 50:
                break
