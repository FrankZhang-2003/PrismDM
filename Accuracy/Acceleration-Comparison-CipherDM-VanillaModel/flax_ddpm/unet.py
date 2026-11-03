import math
import flax
from flax import linen as nn
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax_ddpm.approximation_methods import *
from dataclasses import (
  dataclass,
  field
)

def get_norm(norm, num_groups=None):
    if norm == "in":
        return nn.GroupNorm(num_groups=1)  # InstanceNorm as GroupNorm with 1 group
    elif norm == "bn":
        return nn.BatchNorm(use_running_average=False, momentum=0.9, epsilon=1e-5, dtype=None)
    elif norm == "gn":
        if num_groups is None:
            raise ValueError("num_groups must be specified for group normalization")
        return nn.GroupNorm(num_groups=num_groups)
    elif norm is None:
        return lambda x: x  # No normalization
    else:
        raise ValueError("unknown normalization type")



class TimeMLP(nn.Module):
    base_channels: int
    time_emb_scale: float = 1.0
    time_emb_dim: int = None
    activation: callable = nn.relu

    def setup(self):
        self.pos_emb = PositionalEmbedding(self.base_channels, self.time_emb_scale)
        self.layers_1 = nn.Dense(self.time_emb_dim)
        self.layers_3 = nn.Dense(self.time_emb_dim)

    def __call__(self, x):
        x = self.pos_emb(x) #layer_0
        x = self.layers_1(x) #layer_1
        x = self.activation(x)
        TimeMLP_activation = x
        x = self.layers_3(x)
        return x, TimeMLP_activation



class PositionalEmbedding(nn.Module):
    dim: int
    scale: float

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        assert self.dim % 2 == 0, "Embedding dimension must be even"

        # Create the embedding factors
        emb = math.log(10000) / half_dim
        emb = jnp.exp(jnp.arange(half_dim) * -emb)

        # Scale and compute the sine and cosine embeddings
        emb = jnp.outer(x * self.scale, emb)
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)

        return emb



class Downsample(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x, time_emb=None, y=None):
        if x.shape[-2] % 2 == 1:
            raise ValueError("Downsampling tensor height should be even")
        if x.shape[-1] % 2 == 1:
            raise ValueError("Downsampling tensor width should be even")

        # Define the downsampling convolutional layer
        downsample = nn.Conv(features=self.in_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

        return downsample(x)



class Upsample(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x, time_emb=None, y=None):
        # Upsample using bilinear interpolation and apply a convolution
        upsample = nn.ConvTranspose(features=self.in_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

        return upsample(x)



class AttentionBlock(nn.Module):
    in_channels: int
    norm_type: str
    num_groups: int = 32
    softmax: callable = nn.softmax

    def setup(self):
        self.norm = get_norm(self.norm_type, self.num_groups)
        self.to_qkv = nn.Conv(features=self.in_channels * 3, kernel_size=(1, 1))
        self.to_out = nn.Conv(features=self.in_channels, kernel_size=(1, 1))

    def __call__(self, x):
        b, h, w, c = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = jnp.split(qkv, 3, axis=-1)


        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c).transpose(0, 2, 1)
        v = v.reshape(b, h * w, c)


        dot_products = jnp.matmul(q, k) * (c ** -0.5)
        attention = self.softmax(dot_products, axis=-1)
        out = jnp.matmul(attention, v)
        out = out.reshape(b, h, w, c)
        return self.to_out(out) + x, attention


class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float
    time_emb_dim: int = None
    num_classes: int = None
    norm: str = "gn"
    num_groups: int = 32
    use_attention: bool = False
    activation: list = field(default_factory=list)
    softmax: callable = nn.softmax

    @nn.compact
    def __call__(self, x, time_emb=None, y=None,deterministic=True):
        # Self.activation is replaced as one list and are indexed with current activation number

        # First set of layers
        norm_1 = get_norm(self.norm, self.num_groups)
        conv_1 = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')

        out = self.activation[0](norm_1(x))
        activation_Res1=out
        out = conv_1(out)

        # Time and class conditioning
        if self.time_emb_dim is not None:
            if time_emb is None:
                raise ValueError("Time conditioning was specified but time_emb is not passed")
            time_bias = nn.Dense(features=self.out_channels)
            activation_Res2=self.activation[1](time_emb)
            out += time_bias(activation_Res2)[:, None, None, :]

        if self.num_classes is not None:
            if y is None:
                raise ValueError("Class conditioning was specified but y is not passed")
            class_bias = nn.Embed(self.num_classes, self.out_channels)
            out += class_bias(y)[:, None, None, :]

        # Second set of layers
        norm_2 = get_norm(self.norm, self.num_groups)
        conv_2 = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')

        out = self.activation[2](norm_2(out))
        activation_Res3=out
        out = nn.Dropout(rate=self.dropout,deterministic=deterministic)(out)
        out = conv_2(out)

        # Residual connection
        residual_connection = nn.Conv(self.out_channels, kernel_size=(1, 1)) if self.in_channels != self.out_channels else lambda x: x
        out += residual_connection(x)

        # Attention block
        if self.use_attention:
            attention = AttentionBlock(self.out_channels, self.norm, self.num_groups, self.softmax)
            out, softmax_Res1 = attention(out)
        else:
            attention = lambda x: x
            out = attention(out)
            softmax_Res1=0

        return out, activation_Res1, activation_Res2, activation_Res3, softmax_Res1




class UNet(nn.Module):
    img_channels: int
    base_channels: int
    channel_mults: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_emb_dim: int = None
    time_emb_scale: float = 1.0
    num_classes: int = None
    dropout: float = 0.1
    attention_resolutions: tuple = ()
    norm: str = "gn"
    num_groups: int = 32
    initial_pad: int = 0
    activation: callable = nn.relu
    unlinear_function_dict: dict = field(default_factory=dict)
    def setup(self):
        #activation_judge is now integrated into the whole setup program
        self.time_mlp = None
        if self.time_emb_dim is not None:
            self.time_mlp = TimeMLP(base_channels=self.base_channels, time_emb_scale=self.time_emb_scale, time_emb_dim=self.time_emb_dim, activation=self.unlinear_function_dict['TimeMLP_activation'])
        # print("time_mlp",self.time_mlp)
        self.init_conv = nn.Conv(features=self.base_channels, kernel_size=(3, 3), padding='SAME')

        # Define downsampling and upsampling blocks
        channels = [self.base_channels]
        now_channels = self.base_channels

        downs = []
        for i, mult in enumerate(self.channel_mults):
            out_channels = self.base_channels * mult

            for res_block_pointer in range(self.num_res_blocks):
                current_Down_activation_list=[self.unlinear_function_dict[f"Downs_{res_block_pointer}_activation_Res1"],self.unlinear_function_dict[f"Downs_{res_block_pointer}_activation_Res2"],self.unlinear_function_dict[f"Downs_{res_block_pointer}_activation_Res3"]]
                downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    self.dropout,
                    time_emb_dim=self.time_emb_dim,
                    num_classes=self.num_classes,
                    norm=self.norm,
                    num_groups=self.num_groups,
                    use_attention=i in self.attention_resolutions,
                    activation = current_Down_activation_list,
                    softmax = self.unlinear_function_dict[f"Downs_{res_block_pointer}_softmax_Res1"]
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(self.channel_mults) - 1:
                downs.append(Downsample(now_channels))
                channels.append(now_channels)
        self.downs = downs

        current_Mid_activation_list0=[self.unlinear_function_dict[f"Mids_0_activation_Res1"],self.unlinear_function_dict[f"Mids_0_activation_Res2"],self.unlinear_function_dict[f"Mids_0_activation_Res3"]]
        current_Mid_activation_list1=[self.unlinear_function_dict[f"Mids_1_activation_Res1"],self.unlinear_function_dict[f"Mids_1_activation_Res2"],self.unlinear_function_dict[f"Mids_1_activation_Res3"]]
        self.mid = [
            ResidualBlock(
                now_channels,
                now_channels,
                self.dropout,
                time_emb_dim=self.time_emb_dim,
                num_classes=self.num_classes,
                norm=self.norm,
                num_groups=self.num_groups,
                use_attention=True,
                activation = current_Mid_activation_list0,
                softmax = self.unlinear_function_dict[f"Mids_0_softmax_Res1"]
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                self.dropout,
                time_emb_dim=self.time_emb_dim,
                num_classes=self.num_classes,
                norm=self.norm,
                num_groups=self.num_groups,
                use_attention=False,
                activation = current_Mid_activation_list1,
                softmax = self.unlinear_function_dict[f"Mids_1_softmax_Res1"]
            ),
        ]

        ups=[]
        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_channels = self.base_channels * mult

            for res_block_pointer in range(self.num_res_blocks + 1):
                current_Up_activation_list=[self.unlinear_function_dict[f"Ups_{res_block_pointer}_activation_Res1"],self.unlinear_function_dict[f"Ups_{res_block_pointer}_activation_Res2"],self.unlinear_function_dict[f"Ups_{res_block_pointer}_activation_Res3"]]
                ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    self.dropout,
                    time_emb_dim=self.time_emb_dim,
                    num_classes=self.num_classes,
                    norm=self.norm,
                    num_groups=self.num_groups,
                    use_attention=i in self.attention_resolutions,
                    activation = current_Up_activation_list,
                    softmax = self.unlinear_function_dict[f"Ups_{res_block_pointer}_softmax_Res1"]
                ))
                now_channels = out_channels

            if i != 0:
                ups.append(Upsample(now_channels))
        self.ups = ups

        assert len(channels) == 0

        #Extra Out_Activation Layer during Setup stage
        self.out_activation = self.unlinear_function_dict['Final_activation']
        self.out_norm = get_norm(self.norm, self.num_groups)
        self.out_conv = nn.Conv(features=self.img_channels, kernel_size=(3, 3), padding='SAME')

    def __call__(self, x, time=None, y=None):
        unlinear_dict={}

        ip = self.initial_pad
        if ip != 0:
            x = jnp.pad(x, ((0, 0), (ip, ip), (ip, ip), (0, 0)))

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("Time conditioning was specified but time is not passed")
            time_emb, TimeMLP_activation = self.time_mlp(time)
            unlinear_dict['TimeMLP_activation'] = TimeMLP_activation
        else:
            time_emb = None
        if self.num_classes is not None and y is None:
            raise ValueError("Class conditioning was specified but y is not passed")

        #print(self.time_mlp.activation)

        x = self.init_conv(x)

        skips = [x]

        layer_count=0
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x, activation_Res1, activation_Res2, activation_Res3, softmax_Res1 = layer(x, time_emb, y)
                unlinear_dict[f"Downs_{layer_count}_activation_Res1"] = activation_Res1
                unlinear_dict[f"Downs_{layer_count}_activation_Res2"] = activation_Res2
                unlinear_dict[f"Downs_{layer_count}_activation_Res3"] = activation_Res3
                unlinear_dict[f"Downs_{layer_count}_softmax_Res1"] = softmax_Res1
            else:
                x = layer(x, time_emb, y)
            skips.append(x)

            layer_count=layer_count+1

        layer_count=0
        for layer in self.mid:
            x, activation_Res1, activation_Res2, activation_Res3, softmax_Res1 = layer(x, time_emb, y)
            unlinear_dict[f"Mids_{layer_count}_activation_Res1"] = activation_Res1
            unlinear_dict[f"Mids_{layer_count}_activation_Res2"] = activation_Res2
            unlinear_dict[f"Mids_{layer_count}_activation_Res3"] = activation_Res3
            unlinear_dict[f"Mids_{layer_count}_softmax_Res1"] = softmax_Res1
            layer_count=layer_count+1

        layer_count=0
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = jnp.concatenate([x, skips.pop()], axis=-1)
                x, activation_Res1, activation_Res2, activation_Res3, softmax_Res1 = layer(x, time_emb, y)
                unlinear_dict[f"Ups_{layer_count}_activation_Res1"] = activation_Res1
                unlinear_dict[f"Ups_{layer_count}_activation_Res2"] = activation_Res2
                unlinear_dict[f"Ups_{layer_count}_activation_Res3"] = activation_Res3
                unlinear_dict[f"Ups_{layer_count}_softmax_Res1"] = softmax_Res1
            else:
                x = layer(x, time_emb, y)
            layer_count=layer_count+1

        #Small subsitituion
        x = self.out_activation(self.out_norm(x))
        unlinear_dict[f"Final_activation"] = x
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, ip:-ip, ip:-ip, :], unlinear_dict
        else:
            return x, unlinear_dict
