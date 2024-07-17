import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from FR_config import config
from timm.models.vision_transformer import Block
from torch import nn

from einops import rearrange

from models.FR_vit_small import ViT

class crossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(crossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        b, c, h, w = x.shape
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)

        context = rearrange(context, 'b c h w -> b (h w) c', h=h, w=w)

        Q = self.Wq(x)
        K = self.Wk(context)
        V = self.Wv(context)

        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=1)
        out = torch.einsum('bij,bjd -> bid', att_weights, V)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj_out(out)

        return out

class IQA(nn.Module):
    def __init__(self, embed_dim=768, D1=512, D2=256,
                 img_size=224, patch_size=16, depth=6, heads=16,  num_outputs=1, mlp_dim=2048,
                  **kwargs):
        super().__init__()
        self.D1 = D1
        self.D2 = D2
        self.img_size = img_size
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.num_outputs = 1

        self.input_size = img_size // patch_size

        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        
        self.conv1 = nn.Conv2d(embed_dim, self.D1, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.D1, self.D2, 1, 1, 0)

        self.conv3 = nn.Conv2d(embed_dim * 9, embed_dim * 3, 1, 1, 0)
        self.conv4= nn.Conv2d(embed_dim * 6, embed_dim * 3, 1, 1, 0)

        self.conv5 = nn.Conv2d(self.D1 * 3, self.D1, 1, 1, 0)
        self.conv6 = nn.Conv2d(embed_dim * 3, self.D1, 1, 1, 0)

        self.fc_2 = nn.Sequential(
            nn.Linear(self.D1, self.D2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.D2, self.num_outputs),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(self.D1, self.D2),
            nn.GELU(),
            nn.Linear(self.D2, self.num_outputs),
        )

        self.crossAttention = crossAttention(
            in_channels=512,
            emb_dim=512,
            att_dropout=0.0,
            aropout=0.0
        )

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.Dropout(0.1)
        )

        self.vit1 = ViT(
            image_size=self.img_size,
            patch_size=self.patch_size,
            num_classes=1000,
            dim=self.D1,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.vit2 = ViT(
            image_size=self.img_size,
            patch_size=self.patch_size,
            num_classes=1000,
            dim=self.D2,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )

    def forward(self, x_d, x_r): 
        '''
        # model begin
        '''
        # ref
        ref_x0 = self.vit(x_r[0]).cuda()
        ref_x1 = self.vit(x_r[1]).cuda()
        ref_x2 = self.vit(x_r[2]).cuda()

        ref_x0 = rearrange(ref_x0, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        ref_x1 = rearrange(ref_x1, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        ref_x2 = rearrange(ref_x2, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)  # 4, 768, 14, 14

        ref_x0 = self.conv1(ref_x0)
        ref_x1 = self.conv1(ref_x1)
        ref_x2 = self.conv1(ref_x2)

        # dis
        x0 = self.vit(x_d[0]).cuda()
        x1 = self.vit(x_d[1]).cuda()
        x2 = self.vit(x_d[2]).cuda()

        x0 = rearrange(x0, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)  # 4, 768, 14, 14

        x0 = self.conv1(x0)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        # cross: ref->q, dis->k&v
        cross_0 = self.crossAttention(ref_x0, x0)
        cross_0 = rearrange(cross_0, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        cross_0 = self.net(cross_0)

        cross_1 = self.crossAttention(ref_x1, x1)
        cross_1 = rearrange(cross_1, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        cross_1 = self.net(cross_1)

        cross_2 = self.crossAttention(ref_x2, x2)
        cross_2 = rearrange(cross_2, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        cross_2 = self.net(cross_2)


        # vit_small
        cross_x = torch.cat((cross_0, cross_1, cross_2), dim=2)
        cross_x = rearrange(cross_x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        cross_x = self.conv5(cross_x)
        cross_x = self.vit1(cross_x)  # [b, 512, 14, 14]

        if config.db_name == 'win5':
            score = self.fc_1(cross_x)
        else:
            score = self.fc_2(cross_x)

        return score