from tsai.models.InceptionTimePlus import InceptionTimePlus17x17, InceptionTimePlus
from torch import nn


class InceptionTimePlus17(InceptionTimePlus):
    def __init__(self, c_in, c_out, seq_len):
        super().__init__(c_in, c_out, seq_len, nf=17, depth=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        return self.sigmoid(x)

