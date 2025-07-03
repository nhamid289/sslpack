# This implementation of ViT is modified from https://github.com/microsoft/Semi-supervised-learning
import os
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from timm.layers import DropPath
from timm.layers.helpers import to_2tuple

from wslearn.networks.utils import load_checkpoint


class _PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class _Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class _Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.0,
                 attn_drop=0.0,
                 init_values=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim,
                               num_heads=num_heads,
                               qkv_bias=qkv_bias,
                               attn_drop=attn_drop,
                               proj_drop=drop)
        self.ls1 = _LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=act_layer,
                        drop=drop)
        self.ls2 = _LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class _VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 init_values=None,
                 embed_layer=_PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 block_fn=_Block,
                 **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(dim=embed_dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     init_values=init_values,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=dpr[i],
                     norm_layer=norm_layer,
                     act_layer=act_layer) for i in range(depth)
        ])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.num_features = self.embed_dim
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def extract(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        x = self.extract(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        return self.head(x)


class VisionTransformer(nn.Module):
    """ VisionTransformer
    """

    def __init__(self, img_size, in_channels, patch_size, num_classes,
                 embed_dim, depth, num_heads, **kwargs):
        super().__init__()
        self.model = _VisionTransformer(img_size=img_size,
                                        in_channels=in_channels,
                                        patch_size=patch_size,
                                        num_classes=num_classes,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        **kwargs)

    def forward(self, x):
        return self.model.forward(x)

    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = load_state_dict_from_url(checkpoint_path,
                                                  map_location='cpu')

        orig_state_dict = checkpoint['model']
        new_state_dict = {}
        for key, item in orig_state_dict.items():

            if key.startswith('module'):
                key = '.'.join(key.split('.')[1:])

            if key.startswith('fc') or key.startswith(
                    'classifier') or key.startswith('mlp') or key.startswith(
                        'head'):
                continue

            if key == 'pos_embed':
                posemb_new = self.model.pos_embed.data
                posemb = item
                item = self._resize_pos_embed_vit(posemb, posemb_new)

            new_state_dict[key] = item

        self.model.load_state_dict(new_state_dict, strict=False)

    def _resize_pos_embed_vit(self,
                              posemb,
                              posemb_new,
                              num_tokens=1,
                              gs_new=()):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        # _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
        ntok_new = posemb_new.shape[1]
        if num_tokens:
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[
                0, num_tokens:]
            ntok_new -= num_tokens
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        gs_old = int(math.sqrt(len(posemb_grid)))
        if not len(gs_new):  # backwards compatibility
            gs_new = [int(math.sqrt(ntok_new))] * 2
        assert len(gs_new) >= 2
        # _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                          -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid,
                                    size=gs_new,
                                    mode='bicubic',
                                    align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3,
                                          1).reshape(1, gs_new[0] * gs_new[1],
                                                     -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb


class ViT_Tiny_2(VisionTransformer):
    """ ViT-Tiny
    """

    def __init__(self, image_size, in_channels, num_classes, **kwargs):
        """ Initialise a ViT-Tiny

        Args:
            image_size: The length/width of the image (assumes square images)
            in_channels: The number of input channels
            num_classes: The number of classes in the output layer
        """
        super().__init__(img_size=image_size,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         patch_size=2,
                         embed_dim=192,
                         depth=12,
                         num_heads=3,
                         drop_path_rate=0.1,
                         **kwargs)


class ViT_Small_2(VisionTransformer):
    """ ViT-Small
    """

    def __init__(self, image_size, in_channels, num_classes, **kwargs):
        """ Initialise a ViT-Small

        Args:
            image_size: The length/width of the image (assumes square images)
            in_channels: The number of input channels
            num_classes: The number of classes in the output layer
        """
        super().__init__(img_size=image_size,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         patch_size=2,
                         embed_dim=384,
                         depth=12,
                         num_heads=6,
                         drop_path_rate=0.2,
                         **kwargs)


class ViT_Base_16(VisionTransformer):
    """ ViT-Base  from original paper (https://arxiv.org/abs/2010.11929).
    """

    def __init__(self, image_size, in_channels, num_classes, **kwargs):
        """ Initialise a ViT-Base

        Args:
            image_size: The length/width of the image (assumes square images)
            in_channels: The number of input channels
            num_classes: The number of classes in the output layer
        """
        super().__init__(img_size=image_size,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         patch_size=2,
                         embed_dim=384,
                         depth=12,
                         num_heads=6,
                         drop_path_rate=0.2,
                         **kwargs)