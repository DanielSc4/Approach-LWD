
# imports
from torch.utils.data import Dataset
import numpy as np


class CustomDatasetBinary(Dataset):
    def __init__(self, df, tokenizer, label_col = ''):
        self.text = tokenizer(
            df['text'].to_list(), 
            padding = 'max_length', 
            max_length = 512, 
            truncation = True,
            return_tensors = "pt",
        )
        self.labels = df[label_col].to_list()

    def __len__(self,):
        return len(self.text['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.text['input_ids'][idx],
            # 'token_type_ids': self.text['token_type_ids'][idx],
            'attention_mask': self.text['attention_mask'][idx],
        }, np.float32(self.labels[idx])
    