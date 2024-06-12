import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from utils.config import labels
import torch.nn.functional as F
from transformers import BertConfig


class BertClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassification, self).__init__(config)
        self.num_labels = len(labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)

        return logits


class TextEmbedder(nn.Module):
    def __init__(self, num_classes):
        super(TextEmbedder, self).__init__()
        self.bert = BertModel(BertConfig.from_pretrained('./bert-large-uncased'))
        self.mlp2 = nn.Linear(1024, num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        logits = self.mlp2(pooled_output)
        return logits


class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w_mse = nn.Parameter(torch.tensor(1.0))
        self.w_ce = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
        self.device = device

    def forward(self, embeddings, logits):
        w_mse = torch.clamp(self.w_mse, min=1e-6)
        w_ce = torch.clamp(self.w_ce, min=1e-6)
        centroids = get_centroids(embeddings)
        similarity_matrix = get_cossim(embeddings, centroids)
        sim_matrix = w_mse * similarity_matrix + self.b
        labels = logits.new_zeros(logits.size(0), dtype=torch.long)
        ce_loss = F.cross_entropy(logits, labels)  # 确保这里处理的是多分类问题
        mse_loss = calc_loss(sim_matrix.T)
        alpha = 1.0
        beta = 1.0
        total_loss = alpha * mse_loss / 12 + beta * ce_loss  # 调整损失权重
        return total_loss


def get_centroids(embeddings):
    # centroids = embeddings.mean(dim=1)
    # return centroids
    embeddings = embeddings[0]

    centroid1 = torch.mean(embeddings[:3], dim=0)
    centroid2 = torch.mean(embeddings[3:6], dim=0)
    centroid3 = torch.mean(embeddings[6:], dim=0)
    return centroid1, centroid2, centroid3


def get_centroid(embeddings, model_num, output_num):
    centroid = 0
    for output_id, output in enumerate(embeddings[model_num]):
        if output_id == output_num:
            continue
        centroid = centroid + output
    centroid = centroid / (len(embeddings[model_num]) - 1)
    return centroid


def get_cossim(embeddings, centroids):
    batch_size = embeddings.size(0)
    cossim_list = []
    centroids_norm = [c / torch.norm(c, dim=-1, keepdim=True) for c in centroids]
    centroids_norm = torch.stack(centroids_norm).squeeze()

    for i in range(batch_size):
        emb = embeddings[i]
        emb_norm = emb / torch.norm(emb, dim=1, keepdim=True)
        cossim = torch.mm(emb_norm, centroids_norm.T)
        cossim_list.append(cossim)

    return torch.stack(cossim_list)


def calc_loss(sim_matrix):
    batch_size = sim_matrix.shape[0]
    total_loss = 0
    mse_loss = nn.MSELoss()
    target_matrix_base = -torch.ones(12, 4, device=sim_matrix.device)
    # indices_1 = [(0, 0), (1, 0), (2, 0), (3, 1),(4, 1), (5, 1),(6, 2), (7, 2), (8, 2), (9, 3), (10, 3), (11, 3)]
    indices_1 = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3)]
    for (i, j) in indices_1:
        target_matrix_base[i, j] = 1

    for b in range(batch_size):
        sim_batch = sim_matrix[b]
        target_matrix = target_matrix_base.clone()
        assert sim_batch.shape == (9, 3), "Expected shape (12, 4) for sim_matrix. Got: {}".format(sim_matrix.shape)
        loss = mse_loss(sim_batch, target_matrix)
        total_loss += loss.item()
    return total_loss / batch_size


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return e_x / e_x.sum(dim=1, keepdim=True)


def cross_entropy_loss(sim_matrix):
    y_pred = softmax(sim_matrix)

    y_true = torch.tensor([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=torch.float32).to(sim_matrix.device)

    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    loss = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=-1))

    return loss
