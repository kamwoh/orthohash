import torch
import torch.nn as nn
from torchvision.models import alexnet

from models import register_network


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )


@register_network('alexnet')
class AlexNet(nn.Module):
    def __init__(self,
                 nbit, nclass, pretrained=False, freeze_weight=False,
                 codebook=None,
                 **kwargs):
        super(AlexNet, self).__init__()

        model = alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        for i in range(6):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)

        in_features = model.classifier[6].in_features
        if codebook is None:  # usual CE
            self.ce_fc = nn.Linear(nbit, nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(nbit, nclass, codebook, learn_cent=False)

        self.hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=False),
            nn.BatchNorm1d(nbit, momentum=0.1)
        )

        nn.init.normal_(self.hash_fc[0].weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

        self.extrabit = 0

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)
            for param in self.fc.parameters():
                param.requires_grad_(False)

    def get_backbone_params(self):
        return list(self.features.parameters()) + list(self.fc.parameters())

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, v
