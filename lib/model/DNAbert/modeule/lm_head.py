import torch.nn as nn


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden,2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        lm_logits = self.linear(x[:, 0])
        return lm_logits # (batch_size, 2)



class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        lm_logits = self.linear(x)
        return lm_logits # (batch_size, len, vocab_size)
