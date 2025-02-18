import torch
import math
import copy
from torch import nn
from functools import partial


import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np

import mrcfile
import json
import sys

from copy import deepcopy

from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, Dice


"""
@author: nabin
Combine atom and amino training together
"""


box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundary issues

BATCH_SIZE = 1  # for now
DATALOADERS = 1

data_splits = list()
idx_val_list_atom = list()
idx_val_list_amino = list()
collect_pred_probs_atom= dict()
collect_pred_probs_amino= dict()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_data(dataset_dir, density_map_name):
    data_splits_old = [splits for splits in os.listdir(dataset_dir)]
    for arr in range(len(data_splits_old)):
        data_splits.append(f"{density_map_name}_{arr}.npz")




class CryoData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(data_splits)

    def __getitem__(self, idx):
        cryodata = data_splits[idx].strip("\n")
        loaded_data = np.load(f"{self.root}/{cryodata}")

        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        protein_embeds_torch = torch.tensor(loaded_data['embeddings'])
        return [protein_torch, protein_embeds_torch]
    

def calc_ce_weights_atom(batch):
    y_zeros = (batch == 0.).sum()
    y_ones = (batch == 1.).sum()
    y_two = (batch == 2.).sum()
    y_three = (batch == 3.).sum()
    nSamples = [y_zeros, y_ones, y_two, y_three]
    normedWeights_1 = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = [x + 1e-5 for x in normedWeights_1]
    balance_weights = torch.FloatTensor(normedWeights).to("cuda")
    return balance_weights      

def calc_ce_weights_amino(batch):
    y_zeros = (batch == 0.).sum()
    y_ones = (batch == 1.).sum()
    y_two = (batch == 2.).sum()
    y_three = (batch == 3.).sum()
    y_four = (batch == 4.).sum()
    y_five = (batch == 5.).sum()
    y_six = (batch == 6.).sum()
    y_seven = (batch == 7.).sum()
    y_eight = (batch == 8.).sum()
    y_nine = (batch == 9.).sum()
    y_ten = (batch == 10.).sum()
    y_eleven = (batch == 11.).sum()
    y_twelve = (batch == 12.).sum()
    y_thirteen = (batch == 13.).sum()
    y_fourteen = (batch == 14.).sum()
    y_fifteen = (batch == 15.).sum()
    y_sixteen = (batch == 16.).sum()
    y_seventeen = (batch == 17.).sum()
    y_eighteen = (batch == 18.).sum()
    y_nineteen = (batch == 19.).sum()
    y_twenty = (batch == 20.).sum()
    nSamples = [y_zeros, y_ones, y_two, y_three, y_four, y_five, y_six, y_seven, y_eight, y_nine, y_ten, y_eleven,
                y_twelve,
                y_thirteen, y_fourteen, y_fifteen, y_sixteen, y_seventeen, y_eighteen, y_nineteen, y_twenty]
    normedWeights_1 = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = [x + 1e-5 for x in normedWeights_1]
    balance_weights = torch.FloatTensor(normedWeights).to("cuda")
    return balance_weights


def build_segformer3d_model(config=None):
    model = SegFormer3D(
        in_channels=config["model_parameters"]["in_channels"],
        sr_ratios=config["model_parameters"]["sr_ratios"],
        embed_dims=config["model_parameters"]["embed_dims"],
        patch_kernel_size=config["model_parameters"]["patch_kernel_size"],
        patch_stride=config["model_parameters"]["patch_stride"],
        patch_padding=config["model_parameters"]["patch_padding"],
        mlp_ratios=config["model_parameters"]["mlp_ratios"],
        num_heads=config["model_parameters"]["num_heads"],
        depths=config["model_parameters"]["depths"],
        decoder_head_embedding_dim=config["model_parameters"][
            "decoder_head_embedding_dim"
        ],
        num_classes=config["model_parameters"]["num_classes"],
        decoder_dropout=config["model_parameters"]["decoder_dropout"],
    )
    return model


class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [8, 8, 8, 8],
        num_heads: list = [4, 8, 16, 32],                    # num_heads: list = [1, 2, 5, 8] (original)
        depths: list = [8, 8, 8, 8],                # depths: list = [2, 2, 2, 2] (original)
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 4,
        decoder_dropout: float = 0.2,
    ):
        """
        in_channels: number of the input channels
        img_volume_dim: spatial resolution of the image volume (Depth, Width, Height)
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratios: at which rate increases the projection dim of the hidden_state in the mlp
        num_heads: number of attention heads
        depths: number of attention layers
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channel of the network
        decoder_dropout: dropout rate of the concatenated feature maps

        """
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )
        # decoder takes in the feature maps in the reversed order
        # reversed_embed_dims = embed_dims[::-1]
        # self.segformer_decoder = SegFormerDecoderHead(
            # input_feature_dims=reversed_embed_dims,
            # decoder_head_embedding_dim=decoder_head_embedding_dim,
            # num_classes=num_classes,
            # dropout=decoder_dropout,
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, embeds):
        # embedding the input
   
        x = self.segformer_encoder(x, embeds)
        # # unpacking the embedded features generated by the transformer
        c1 = x[0]
        c2 = x[1]
        c3 = x[2]
        c4 = x[3]
        # decoding the embedded features
        return [c1, c2, c3, c4]
    
# ----------------------------------------------------- encoder -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        """
        in_channels: number of the channels in the input volume
        embed_dim: embedding dimmesion of the patch
        """
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # standard embedding patch
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        # embedding dimesion of each attention head
        self.attention_head_dim = embed_dim // num_heads

        # The same input is used to generate the query, key, and value,
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, attention_head_size)
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (batch_size, num_patches, hidden_size)
        B, N, C = x.shape

        # (batch_size, num_head, sequence_length, embed_dim)
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )
       

        if self.sr_ratio > 1:
            n = cube_root(N)
            # (batch_size, sequence_length, embed_dim) -> (batch_size, embed_dim, patch_D, patch_H, patch_W)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n)
            # (batch_size, embed_dim, patch_D, patch_H, patch_W) -> (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio) -> (batch_size, sequence_length, embed_dim)
            # normalizing the layer
            x_ = self.sr_norm(x_)
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)
        else:
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)

        k, v = kv[0], kv[1]

        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.2,
        proj_dropout: float = 0.2,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        mlp_ratio: at which rate increasse the projection dim of the embedded patch in the _MLP component
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.embd_mlp = nn.Linear(in_features=2560, out_features=embed_dim )
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x, embed):
        embed = self.embd_mlp(embed).unsqueeze(1)
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x + embed
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
            


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [48, 48, 48, 48],
        depths: list = [24, 24, 24, 24],
    ):
        """
        in_channels: number of the input channels
        img_volume_dim: spatial resolution of the image volume (Depth, Width, Height)
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratio: at which rate increasse the projection dim of the hidden_state in the mlp
        num_heads: number of attenion heads
        depth: number of attention layers
        """
        super().__init__()

        # patch embedding at different Pyramid level
        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        # block 1
        self.tf_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    sr_ratio=sr_ratios[0],
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # block 2
        self.tf_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    sr_ratio=sr_ratios[1],
                    qkv_bias=True,
                )
                for _ in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # block 3
        self.tf_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    sr_ratio=sr_ratios[2],
                    qkv_bias=True,
                )
                for _ in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # block 4
        self.tf_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    sr_ratio=sr_ratios[3],
                    qkv_bias=True,
                )
                for _ in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])
        self.atom_conv = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=1)

    def forward(self, x, embeds):
        out = []
        # at each stage these are the following mappings:
        # (batch_size, num_patches, hidden_state)
        # (num_patches,) -> (D, H, W)
        # (batch_size, num_patches, hidden_state) -> (batch_size, hidden_state, D, H, W)

        # stage 1
        x = self.embed_1(x)

        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block1):
            x = blk(x, embeds)
        x = self.norm1(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 2
        x = self.embed_2(x)
       

        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block2):
            x = blk(x, embeds)
        x = self.norm2(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 3
        x = self.embed_3(x)

        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block3):
            x = blk(x, embeds)
        x = self.norm3(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 4
        x = self.embed_4(x)

        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block4):
            x = blk(x, embeds)
        x = self.norm4(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

    
        return out


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.2):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # added batchnorm (remove it ?)
        # self.bn = nn.BatchNorm3d(dim)
        self.bn = nn.InstanceNorm3d(dim)

    def forward(self, x):
        B, N, C = x.shape
        # (batch, patch_cube, hidden_size) -> (batch, hidden_size, D, H, W)
        # assuming D = H = W, i.e. cube root of the patch is an integer number!
        n = cube_root(N)
        x = x.transpose(1, 2).view(B, C, n, n, n)
        x = self.dwconv(x)
        # added batchnorm (remove it ?)
        # x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x

###################################################################################
def cube_root(n):
    return round(math.pow(n, (1 / 3)))
    

###################################################################################
# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        return x


###################################################################################
class SegFormerDecoderHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
        self,
        input_feature_dims: list = [256, 160, 64, 32],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.2,

    ):
        """
        input_feature_dims: list of the output features channels generated by the transformer encoder
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channels
        dropout: dropout rate of the concatenated feature maps
        """
        super().__init__()
        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )
        # convolution module to combine feature maps generated by the mlps
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.InstanceNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        # final linear projection layer
        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim, num_classes, kernel_size=1
        )

        # segformer decoder generates the final decoded feature map size at 1/4 of the original input volume size
        self.upsample_volume = nn.Upsample(
            scale_factor=4.0, mode="trilinear", align_corners=False
        )

    def forward(self, c1, c2, c3, c4, charlie=None):
       ############## _MLP decoder on C1-C4 ###########
        n, _, _, _, _ = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
            .contiguous()
        )
        _c4 = torch.nn.functional.interpolate(
            _c4,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
            .contiguous()
        )
        _c3 = torch.nn.functional.interpolate(
            _c3,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
            .contiguous()
        )
        _c2 = torch.nn.functional.interpolate(
            _c2,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
            .contiguous()
        )

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        if charlie is not None:
            _c = _c + charlie
        

        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        x = self.upsample_volume(x)

        return x, _c



class MultiTaskCryoModel(pl.LightningModule):
    def __init__(self,learning_rate=1e-4, mode=None):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode

        self.model = SegFormer3D()

        self.segformer_decoder_atom = SegFormerDecoderHead(num_classes=4)
        self.segformer_decoder_amino = SegFormerDecoderHead(num_classes=21)

    def forward(self, density_map_data, esm_embeds):
        """
        Optional forward pass that can be used for inference if needed.
        """
        y_hat = self.model(density_map_data, esm_embeds)
        y_hat_atom, tango = self.segformer_decoder_atom(y_hat[0], y_hat[1], y_hat[2], y_hat[3], charlie=None)

        if self.mode == 'atom':
            return y_hat_atom
        elif self.mode == 'amino':
            y_hat_amino, _ = self.segformer_decoder_amino(y_hat[0], y_hat[1], y_hat[2], y_hat[3], charlie=tango)
            return y_hat_amino
        else:
            print("Pass in either 'atom' or 'amino' in mode.")
            exit()

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
    
        density_map_data, esm_embeds = batch[0], batch[1]
        esm_embeds = esm_embeds.float()
        density_map_data = torch.unsqueeze(density_map_data, 1)

        y_hat_pred = self(density_map_data, esm_embeds)

        probs = torch.softmax(y_hat_pred[0], dim=0)
        
        probs_permute = torch.permute(probs, (1, 2, 3, 0))
 
        vals = torch.argmax(y_hat_pred[0], dim=0)
        
        if self.mode == 'atom':
            idx_val_np_atom = np.empty(shape=(32, 32, 32), dtype='S30')

            for i in range(len(probs_permute)):
                for j in range(len(probs_permute[i])):
                    for k in range(len(probs_permute[i][j])):
                        val_prob = probs_permute[i][j][k]
                        collect_pred_probs_atom[f'{batch_idx}_{i}_{j}_{k}'] = val_prob
                        v = f'{batch_idx}_{i}_{j}_{k}'
                        idx_val_np_atom[i][j][k] = v
            idx_val_list_atom.append(idx_val_np_atom)
        
        else:
            idx_val_np_amino = np.empty(shape=(32, 32, 32), dtype='S30')

            for i in range(len(probs_permute)):
                for j in range(len(probs_permute[i])):
                    for k in range(len(probs_permute[i][j])):
                        val_prob = probs_permute[i][j][k]
                        collect_pred_probs_amino[f'{batch_idx}_{i}_{j}_{k}'] = val_prob
                        v = f'{batch_idx}_{i}_{j}_{k}'
                        idx_val_np_amino[i][j][k] = v
            idx_val_list_amino.append(idx_val_np_amino)
        
        return vals

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser



def infer_classifier(density_map_splits_dir, input_data_dir, density_map_name, amino_checkpoint, atom_checkpoint, infer_run_on, infer_on_gpu):
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MultiTaskCryoModel.add_model_specific_args(parser)

    prepare_data(dataset_dir=density_map_splits_dir, density_map_name=density_map_name)
    dataset = CryoData(density_map_splits_dir)
    test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                             num_workers=1)

    args, unknown = parser.parse_known_args()
    args.detect_anomaly=True
    args.enable_model_summary = True
    if infer_run_on == "gpu":
        args.accelerator = "gpu"
        args.devices = [infer_on_gpu]
    else:
        args.accelerator = "cpu"

    atom_model = MultiTaskCryoModel.load_from_checkpoint(atom_checkpoint, mode='atom')
    trainer = pl.Trainer.from_argparse_args(args)
    atom_predictions = trainer.predict(atom_model, dataloaders=test_loader)

    # Load the amino acid model and run inference
    amino_model = MultiTaskCryoModel.load_from_checkpoint(amino_checkpoint,  mode='amino')
    amino_predictions = trainer.predict(amino_model, dataloaders=test_loader)
    trainer = pl.Trainer.from_argparse_args(args)

    
    for pred in range(len(atom_predictions)):
        atom_predictions[pred] = atom_predictions[pred].numpy()
        
    for pred in range(len(amino_predictions)):
        amino_predictions[pred] = amino_predictions[pred].numpy()


    org_map = f"{input_data_dir}/{density_map_name}/emd_normalized_map.mrc"
    org_map = mrcfile.open(org_map, mode='r')

    recon, idx_val_mat = reconstruct_map(manifest=atom_predictions, idx_val_np=idx_val_list_atom, image_shape=org_map.data.shape)
    filename = "atom_predicted.mrc"
    outfilename = f"{input_data_dir}/{density_map_name}/{density_map_name}_{filename}"
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(recon)
        mrc.voxel_size = 1
        mrc.header.origin = org_map.header.origin
        mrc.close()

    # save the probabilities
    file_prob = f"{input_data_dir}/{density_map_name}/{density_map_name}_probabilities_atom.txt"
    save_probs(outfilename, idx_val_mat, file_prob, mode='atom')


    recon, idx_val_mat = reconstruct_map(manifest=amino_predictions, idx_val_np=idx_val_list_amino, image_shape=org_map.data.shape)
    filename = "amino_predicted.mrc"
    outfilename = f"{input_data_dir}/{density_map_name}/{density_map_name}_{filename}"
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(recon)
        mrc.voxel_size = 1
        mrc.header.origin = org_map.header.origin
        mrc.close()

    # save the probabilities
    file_prob = f"{input_data_dir}/{density_map_name}/{density_map_name}_probabilities_amino.txt"
    save_probs(outfilename, idx_val_mat, file_prob, mode='amino')



def reconstruct_map(manifest, idx_val_np, image_shape):
    # takes the output of model and constructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimentions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimentions[0], dimentions[1], dimentions[2]))

    idx_val_mat = np.empty(shape=(dimentions[0], dimentions[1], dimentions[2]), dtype='S30')

    counter = 0
    for z_steps in range(int(dimentions[2] / core_size)):
        for y_steps in range(int(dimentions[1] / core_size)):
            for x_steps in range(int(dimentions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                idx_val_mat[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    idx_val_np[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                counter += 1
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    idx_val_np_mat = idx_val_mat[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image, idx_val_np_mat



def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions


def get_xyz(idx, voxel, origin):
    return (idx * voxel) + origin


def save_probs(mrc_file, idx_file, file_prob, mode):
    if mode == 'atom':

        mrc_map = mrcfile.open(mrc_file, mode='r')
        x_origin = mrc_map.header.origin['x']
        y_origin = mrc_map.header.origin['y']
        z_origin = mrc_map.header.origin['z']
        x_voxel = mrc_map.voxel_size['x']
        y_voxel = mrc_map.voxel_size['y']
        z_voxel = mrc_map.voxel_size['z']
        mrc_data = deepcopy(mrc_map.data)
        with open(file_prob, "w") as f:
            for k in range(len(mrc_data[2])):
                for j in range(len(mrc_data[1])):
                    for i in range(len(mrc_data[0])):
                        try:
                            if mrc_data[i][j][k] > 0:
                                ids = idx_file[i][j][k]
                                x = round(get_xyz(k, x_voxel, x_origin), 3)
                                y = round(get_xyz(j, y_voxel, y_origin), 3)
                                z = round(get_xyz(i, z_voxel, z_origin), 3)
                                ids = ids.decode()
                                value = collect_pred_probs_atom[ids]
                                lst = value.tolist()
                                lst.insert(0,[x,y,z])
                                json_dump = json.dumps(lst)
                                final = json_dump[1:-1]
                                f.writelines(final)
                                f.writelines('\n')
                        except UnicodeDecodeError:
                            print("Error", i, j, k)
                            pass
                        except IndexError:
                            pass
    else:
        mrc_map = mrcfile.open(mrc_file, mode='r')
        x_origin = mrc_map.header.origin['x']
        y_origin = mrc_map.header.origin['y']
        z_origin = mrc_map.header.origin['z']
        x_voxel = mrc_map.voxel_size['x']
        y_voxel = mrc_map.voxel_size['y']
        z_voxel = mrc_map.voxel_size['z']
        mrc_data = deepcopy(mrc_map.data)
        with open(file_prob, "w") as f:
            for k in range(len(mrc_data[2])):
                for j in range(len(mrc_data[1])):
                    for i in range(len(mrc_data[0])):
                        try:
                            if mrc_data[i][j][k] > 0:
                                ids = idx_file[i][j][k]
                                x = round(get_xyz(k, x_voxel, x_origin), 3)
                                y = round(get_xyz(j, y_voxel, y_origin), 3)
                                z = round(get_xyz(i, z_voxel, z_origin), 3)
                                ids = ids.decode()
                                value = collect_pred_probs_amino[ids]
                                lst = value.tolist()
                                lst.insert(0,[x,y,z])
                                json_dump = json.dumps(lst)
                                final = json_dump[1:-1]
                                f.writelines(final)
                                f.writelines('\n')
                        except UnicodeDecodeError:
                            print("Error", i, j, k)
                            pass
                        except IndexError:
                            pass





if __name__ == "__main__":

    density_map_splits_dir = sys.argv[1]
    input_data_dir = sys.argv[2]
    density_map = sys.argv[3]
    amino_checkpoint = sys.argv[4]
    atom_checkpoint = sys.argv[5]
    infer_run_on = sys.argv[6]
    infer_run_gpu = int(sys.argv[7])
    infer_classifier(density_map_splits_dir=density_map_splits_dir, input_data_dir=input_data_dir, density_map_name=density_map, 
                          amino_checkpoint=amino_checkpoint, atom_checkpoint=atom_checkpoint, infer_run_on=infer_run_on, infer_on_gpu=infer_run_gpu)