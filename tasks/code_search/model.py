import torch
import torch.nn as nn


class OpCodeModel(nn.Module):
    def __init__(self, encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def forward(self, inputs=None, attention_mask=None, position_ids=None):
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[0]
        outputs = (outputs * inputs.ne(1)[:, :, None]).sum(1) / inputs.ne(1).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
