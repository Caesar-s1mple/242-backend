import torch
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device2 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
try:
    import torch_npu
    device = 'npu:5' if torch_npu.npu.is_available() else device
    # device2 = 'npu:6' if torch_npu.npu.is_available() else device
except:
    pass

try:
    a = torch.tensor(1, device=device)
    del a
except:
    device = 'cpu'

with open('./utils/labels.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)['labels']
