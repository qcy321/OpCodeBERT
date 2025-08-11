import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel


class OpCodeBERTClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classification = ClassificationHead(config)
        self.post_init()

    def forward(self, inputs, labels):
        outputs = self.roberta(inputs, attention_mask=inputs.ne(1))[1]
        outputs = self.classification(outputs)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        return loss, outputs


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
