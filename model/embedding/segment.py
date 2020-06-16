import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(7, embed_size, padding_idx=0)
