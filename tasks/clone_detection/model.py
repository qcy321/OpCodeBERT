import torch
from transformers import RobertaPreTrainedModel, RobertaModel


class OpCodeBERTCloneDetection(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(self, inputs, labels=None):
        bs, _ = inputs[0].size()
        inputs = torch.cat((inputs[0], inputs[1], inputs[2]), 0)
        e = 20

        outputs = self.roberta(inputs, attention_mask=inputs.ne(1))[0]
        outputs = (outputs * inputs.ne(1)[:, :, None]).sum(1) / inputs.ne(1).sum(1)[:, None]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)

        outputs = outputs.split(bs, 0)

        prob_1 = (outputs[0] * outputs[1]).sum(-1) * e
        prob_2 = (outputs[0] * outputs[2]).sum(-1) * e
        temp = torch.cat((outputs[0], outputs[1]), 0)
        temp_labels = torch.cat((labels, labels), 0)
        prob_3 = torch.mm(outputs[0], temp.t()) * e
        mask = labels[:, None] == temp_labels[None, :]
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()
        return loss, outputs[0]
