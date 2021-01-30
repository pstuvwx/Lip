import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, ch, ksize):
        super(ResBlock, self).__init__()
        self.c1 = nn.Conv1d(ch, ch, ksize, 1, ksize//2)
        self.c2 = nn.Conv1d(ch, ch, ksize, 1, ksize//2)

        self.relu = nn.ReLU()


    def forward(self, x):
        h = x
        h = self.relu(h)
        h = self.c1(h)
        h = self.relu(h)
        h = self.c2(h)
        h = h + x
        return h



def ConvModel(n_frq, n_class, n_unit, n_layer, ksize):
    dst = nn.Sequential(
        nn.Conv1d(n_frq,  n_unit,  ksize, 1, ksize//2),
        *[ResBlock(n_unit, ksize) for _ in range(n_layer)],
        nn.Conv1d(n_unit, n_class, ksize, 1, ksize//2),
    )
    return dst
