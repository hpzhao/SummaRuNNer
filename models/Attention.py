import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the query features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*query \\
            attn_scores = exp(x_i) / sum_j exp(x_j) \\
            attn_out = attn * context
            \end{array}

    Args:
        dim(int): The number of expected features in the query

    Inputs: query, context
        - **query** (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: query, attn
        - **query** (batch, query_len, dimensions): tensor containing the attended query features from the decoder.
        - **attn** (batch, query_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    """
    def __init__(self):
        super(Attention, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask
    
    """
        - query   (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - context (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    """
    def forward(self, query, context):
        batch_size = query.size(0)
        dim = query.size(2)
        in_len = context.size(1)
        # (batch, query_len, dim) * (batch, in_len, dim) -> (batch, query_len, in_len)
        attn = torch.bmm(query, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn_scores = F.softmax(attn.view(-1, in_len),dim=1).view(batch_size, -1, in_len)

        # (batch, query_len, in_len) * (batch, in_len, dim) -> (batch, query_len, dim)
        attn_out = torch.bmm(attn_scores, context)

        return attn_out, attn_scores

if __name__ == '__main__':
    torch.manual_seed(1)
    attention = Attention()
    context = Variable(torch.randn(10, 20, 4))
    query = Variable(torch.randn(10, 1, 4))
    query, attn = attention(query, context)
    print(query)
