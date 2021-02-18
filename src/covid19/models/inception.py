from tsai.models.InceptionTime import InceptionTime


class InceptionTimeEx(InceptionTime):
    def __init__(self, c_in, c_out, seq_len):
        super().__init__(c_in, c_out)
