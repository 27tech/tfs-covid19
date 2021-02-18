from tsai.models.TransformerModel import TransformerModel


class Transformer(TransformerModel):
    def __init__(self, c_in, c_out, seq_len):
        super().__init__(c_in=c_in, c_out=c_out)