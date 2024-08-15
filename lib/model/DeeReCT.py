from collections import namedtuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class DeeReCTConfig:
    d_version: int = 1.0


class DeeReCT(nn.Module):
    def __init__(self, vocab_size, device=None, dtype=None,) -> None:
        super(DeeReCT, self).__init__()
        self.config = DeeReCTConfig()
        self.vocab_size = vocab_size
        factory_kwargs = {"device": device, "dtype": dtype}

        self.conv1_1 = nn.Conv1d(in_channels= vocab_size, out_channels=4, kernel_size=15, stride=1, padding='same', **factory_kwargs)
        self.conv1_2 = nn.Conv1d(in_channels= 4, out_channels=4, kernel_size=15, stride=1, padding='same', **factory_kwargs)
        self.relu = nn.ReLU()

        self.conv2_1 = nn.Conv1d(in_channels= vocab_size, out_channels=4, kernel_size=1, stride=1, padding='same', **factory_kwargs)
        self.conv2_2 = nn.Conv1d(in_channels= 4, out_channels=4, kernel_size=1, stride=1, padding='same', **factory_kwargs)
        self.pool = nn.AvgPool1d(2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(601*(4+2), 2, **factory_kwargs)
       
    def forward(self, input_ids):
        input = F.one_hot(input_ids, num_classes=self.vocab_size).float().transpose(-1,-2) # (batch, 601, 4)

        x1_1 = self.relu(self.conv1_1(input)) # (batch, 4, 601) 
        x1_2 = self.relu(self.conv1_2 (x1_1)) # (batch, 4, 601) 
        x1 = self.flatten(x1_2) # # (batch, 4* 601) 
        output1 = self.dropout(x1)

        x2_1 = self.relu(self.conv2_1(input)) # (batch, 4, 601) 
        x2_2 = self.relu(self.conv2_2(x2_1))  # (batch, 4, 601) 
        x2 = self.flatten(self.pool(x2_2.transpose(-1,-2))) # (batch, 601*2)  
        output2 = self.dropout(x2)

        output = torch.cat((output1, output2), -1)  
        lm_logits = self.fc(output)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)        


    @classmethod
    def from_funetuned(cls, pretrained_model_name, vocab_size, device=None, dtype=None,):
        model = cls(vocab_size, device=device, dtype=dtype)
        model.load_state_dict(torch.load(pretrained_model_name))
        print("loading successful:", pretrained_model_name)
        return model


