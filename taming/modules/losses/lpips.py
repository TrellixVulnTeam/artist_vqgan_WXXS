"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, content_input, target, return_feats=False):
        input_c, input_t = self.scaling_layer(content_input), self.scaling_layer(target)
        outs_c, outs_t = self.net(input_c), self.net(input_t)
        feats_c, feats_t, diffs = list(), list(), list()
        for kk in range(len(self.chns)):
            feats_c.append(normalize_tensor(outs_c[kk]))
            feats_t.append(normalize_tensor(outs_t[kk]))
            diffs.append((feats_c[-1] - feats_t[-1]) ** 2)

        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]

        if return_feats:
            return val.mean(), outs_c, outs_t
        return val.mean()


class LPIPSWithStyle(LPIPS):
    def __init__(self, use_dropout=True):
        super(LPIPSWithStyle, self).__init__(use_dropout=use_dropout)
        self.style_loss = nn.MSELoss(reduce=False)

    def forward(self, content_input, target, style_input=None):
        print()
        print('input: ', content_input.shape, target.shape)
        loss_c, outs_c, outs_t = super().forward(content_input, target, return_feats=True)

        outs_s = self.net(self.scaling_layer(style_input)) if style_input is not None else outs_c
        diffs = list()
        for kk in range(len(self.chns)):
            smooth_out_s = double_softmax(outs_s[kk])
            smooth_out_t = double_softmax(outs_t[kk])
            print('compare smooth: ', outs_s[kk].shape, smooth_out_s.shape)
            std_s, mean_s = self.calc_mean_std(smooth_out_s)
            std_t, mean_t = self.calc_mean_std(smooth_out_t)
            diff = self.style_loss(std_s, std_t) + self.style_loss(mean_s, mean_t)
            print('diff: ', diff.shape)
            print('loss weight: ', self.calc_balanced_loss_scale(smooth_out_s, smooth_out_t).shape)
            diff = diff * self.calc_balanced_loss_scale(smooth_out_s, smooth_out_t)
            diffs.append(diff.sum())

        val = diffs[0]
        for l in range(1, len(self.chns)):
            val += diffs[l]
        print('content, style loss: ', loss_c.detach().item(), val.detach().item())
        return loss_c, val

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        N, *_ = feat.shape
        feat_var = feat.view(N, -1).var(dim=-1) + eps
        feat_std = feat_var.sqrt().view(N, 1, 1, 1)
        print('std: ', feat_std.shape)
        feat_mean = feat.view(N, -1).mean(dim=-1).view(N, 1, 1, 1)
        print('mean: ', feat_std.shape)
        return feat_mean, feat_std

    @staticmethod
    def calc_balanced_loss_scale(feat, target_feat):
        return torch.mean(feat ** 2, dim=(1, 2, 3)) + torch.mean(target_feat ** 2, dim=(1, 2, 3))


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def double_softmax(x, eps=1e-10):
    exp_x = torch.exp(x)
    return exp_x / (exp_x.sum(dim=(-2, -1), keepdim=True) + eps)
