
from typing import Tuple
import torch
import torch.nn as nn

HIDDEN_SIZE = 32
VOCAB_SIZE =196906
EMBEDDING_DIM = 64 # embedding_dim 
SEQ_LEN = 100
BATCH_SIZE = 64


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()

    def forward(
        self,
        lstm_outputs: torch.Tensor,  # BATCH_SIZE x SEQ_LEN x HIDDEN_SIZE
        final_hidden: torch.Tensor,  # BATCH_SIZE x HIDDEN_SIZE
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Bahdanau Attention module

        Args:
            keys (torch.Tensor): lstm hidden states (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
            query (torch.Tensor): lstm final hidden state (BATCH_SIZE, HIDDEN_SIZE)

        Returns:
            Tuple[torch.Tensor]:
                context_matrix (BATCH_SIZE, HIDDEN_SIZE)
                attention scores (BATCH_SIZE, SEQ_LEN)
        """
        # input:
        # keys – lstm hidden states (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # query - lstm final hidden state (BATCH_SIZE, HIDDEN_SIZE)

        keys = self.W_k(lstm_outputs)
        # print(f'After linear keys: {keys.shape}')

        query = self.W_q(final_hidden)
        # print(f"After linear query: {query.shape}")

        # print(f"query.unsqueeze(1) {query.unsqueeze(1).shape}")

        sum = query.unsqueeze(1) + keys
        # print(f"After sum: {sum.shape}")

        tanhed = self.tanh(sum)
        # print(f"After tanhed: {tanhed.shape}")

        vector = self.W_v(tanhed).squeeze(-1)
        # print(f"After linear vector: {vector.shape}")

        att_weights = torch.softmax(vector, -1)
        # print(f"After softmax att_weights: {att_weights.shape}")

        context = torch.bmm(att_weights.unsqueeze(1), keys).squeeze()
        # print(f"After bmm context: {context.shape}")

        return context, att_weights

        # att_weights = self.linear(lstm_outputs)
        # # print(f'After linear: {att_weights.shape, final_hidden.unsqueeze(2).shape}')

        # att_weights = self.linear(lstm_outputs)
        # # print(f'After linear: {att_weights.shape, final_hidden.unsqueeze(2).shape}')
        # att_weights = torch.bmm(att_weights, final_hidden.unsqueeze(2))
        # # print(f'After bmm: {att_weights.shape}')
        # att_weights = F.softmax(att_weights.squeeze(2), dim=1)
        # # print(f'After softmax: {att_weights.shape}')
        # cntxt = torch.bmm(lstm_outputs.transpose(1, 2), att_weights.unsqueeze(2))
        # # print(f'Context: {cntxt.shape}')
        # concatted = torch.cat((cntxt, final_hidden.unsqueeze(2)), dim=1)
        # # print(f'Concatted: {concatted.shape}')
        # att_hidden = self.tanh(self.align(concatted.squeeze(-1)))
        # # print(f'Att Hidden: {att_hidden.shape}')
        # return att_hidden, att_weights

# Test on random numbers
BahdanauAttention()(torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), torch.randn(BATCH_SIZE, HIDDEN_SIZE))[1].shape


class LSTMConcatAttentionEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        # self.embedding = embedding_layer
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.attn = BahdanauAttention(HIDDEN_SIZE)
        self.clf = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), 
            nn.Dropout(), 
            nn.Tanh(), 
            nn.Linear(128, 1)
        )

    def forward(self, x):        
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(att_hidden)
        return out, att_weights
    
    