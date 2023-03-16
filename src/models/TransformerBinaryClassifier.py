import torch
import torch.nn as nn

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, model, dropout_rate = .2):
        super(TransformerBinaryClassifier, self).__init__()

        self.lm = model
        self.dropout = nn.Dropout(dropout_rate)
        # self.relu = nn.ReLU()
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, mask,):
        _, pooled_output = self.lm(
            input_ids = input_ids,
            attention_mask = mask,
            return_dict = False
        )
        x = self.dropout(pooled_output)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x