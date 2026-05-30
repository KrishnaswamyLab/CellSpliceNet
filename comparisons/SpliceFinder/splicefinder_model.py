import torch.nn as nn


class SpliceFinder(nn.Module):
    '''
    SpliceFinder: ab initio prediction of splice sites using convolutional neural network.
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3306-3
    '''
    def __init__(self,
                 in_channels: int = 1,
                 num_features: int = 32,
                 seq_len: int = 4096):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=num_features, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(seq_len * num_features, 100),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        out = self.model(x)
        return out

