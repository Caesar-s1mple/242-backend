import torch
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
try:
    a = torch.tensor(1, device=device)
    del a
except:
    device = 'cpu'

with open('./utils/labels.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)['labels']
