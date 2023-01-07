# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import sys
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv1x1 = layers.ddpm_conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points
        self.k = 3

        #改良後
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        ##print("InputTNet開始!") input_data.shape == (16, 10000, 3)

        #改良後
        h = F.relu(self.bn1(self.conv1(input_data)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        pool = nn.MaxPool1d(h.size(-1))(h)
        ##print("pool", pool.shape)
        flat = nn.Flatten(1)(pool) #(16, 9216)

        h = F.relu(self.bn4(self.fc1(flat)))
        h = F.relu(self.bn5(self.fc2(h)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(16,1,1)
        if h.is_cuda:
            init=init.cuda()
        matrix = self.fc3(h).view(-1,self.k,self.k) + init
        output = torch.bmm(torch.transpose(input_data,1,2), matrix).transpose(1,2)
        return output



class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points
        self.k = 64

        #改良後
        self.conv1 = nn.Conv1d(64,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,4096)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        #改良後
        ##print("FeatureTNet開始!")
        ##print(input_data.shape)
        #改良後
        h = F.relu(self.bn1(self.conv1(input_data)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        pool = nn.MaxPool1d(h.size(-1))(h)
        flat = nn.Flatten(1)(pool)

        h = F.relu(self.bn4(self.fc1(flat)))
        h = F.relu(self.bn5(self.fc2(h)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(16,1,1)
        if h.is_cuda:
            init=init.cuda()
        matrix = self.fc3(h).view(-1,self.k,self.k) + init
        output = torch.bmm(torch.transpose(input_data,1,2), matrix).transpose(1,2)
        return output

"""ポイントネット"""
class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.num_points = config.model.num_points
        self.num_channels = config.data.num_channels
        self.nf = nf = config.model.nf
        dropout = config.model.dropout # 0.1
        self.act = act = get_act(config)
        print(self.act)
        # Condition on noise levels.
        modules = [nn.Linear(nf, nf * 4)]
        modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
        nn.init.zeros_(modules[0].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)

        nn.init.zeros_(modules[1].bias)
        self.InputTNet = InputTNet(self.num_points)
        self.FeatureTNet = FeatureTNet(self.num_points)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv1_1 = nn.Conv1d(64,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 3, 1)

        self.resNet1 = ResnetBlock(64, 64)

        self.resNet2 = ResnetBlock(64, 64)
        self.resNet3 = ResnetBlock(128, 128)

        self.resNet4 = ResnetBlock(512, 512)
        self.resNet5 = ResnetBlock(256, 256)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(3)

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t):
      modules = self.all_modules
      #改良後
      ##print(x.shape) == (16, 3, 10000)
      timesteps = t
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[0](temb)
      temb = modules[2](self.act(temb))

      h = F.relu(self.bn1(self.conv1(self.InputTNet(x))))
      h = self.resNet1(h, temb)
      ##print(281, h.shape)
      h = F.relu(self.bn1_1(self.conv1_1(h)))
      ##print(283, h.shape)
      h = self.FeatureTNet(h)
      h_64 = h
      h = self.resNet2(h, temb)
      h = F.relu(self.bn2(self.conv2(h)))
      h = self.resNet3(h, temb)
      h = self.bn3(self.conv3(h))
      #print(288, h.shape)
      global_vector = nn.MaxPool1d(h.size(-1))(h)
      #print(290, global_vector.shape)
      #global_vector = nn.Flatten(1)(h)
      #print(292, global_vector.shape)
      #print(295, global_vector.shape)
      global_vector_pre = torch.cat((global_vector, global_vector), dim=2)
      global_vector_pre = torch.cat((global_vector_pre, global_vector_pre), dim=2)
      global_vector_pre = torch.cat((global_vector_pre, global_vector_pre), dim=2)
      global_vector_pre = torch.cat((global_vector_pre, global_vector_pre), dim=2)
      #print(300, global_vector_pre.shape)
      ##print("h8",h8.shape)
      #まずh8を(1024,1)から(1024,10000)に変更する
      while global_vector.shape[2] < 8000:
        global_vector = torch.cat((global_vector, global_vector), dim=2)
        ##print("h8",h8.shape)
      while global_vector.shape[2] < 10000:
        global_vector = torch.cat((global_vector, global_vector_pre), dim=2)
      #print("308", global_vector.shape)
      #h4にh8を結合
      h = torch.cat((h_64, global_vector), dim=1) #(1088, 10000)

      h = F.relu(self.bn4(self.conv4(h)))
      h = self.resNet4(h, temb)
      h = F.relu(self.bn5(self.conv5(h)))
      h = self.resNet5(h, temb)
      h = F.relu(self.bn6(self.conv6(h)))
      h = self.bn7(self.conv7(h))

      return h
      
      
"""
class DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf # 32
    ch_mult = config.model.ch_mult # (1, 2, 2, 2)
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks #2
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions #(16,)
#    ####print('attn_resolutions=',attn_resolutions)
    dropout = config.model.dropout # 0.1
    resamp_with_conv = config.model.resamp_with_conv # True
    self.num_resolutions = num_resolutions = len(ch_mult)# 4
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional # True
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional: #True
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered #False
    channels = config.data.num_channels #1

    # Downsampling block
    modules.append(conv1x1(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):#8
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv1x1(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, labels):
    modules = self.all_modules
    ###print("114行 : モジュールたちは\n",modules)
    
    m_idx = 0
    if self.conditional: #True
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None


    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.
    # Downsampling block
    ####print("137行 : hの形は", h.shape)
    h = modules[m_idx](h)
    hs = [h]
    m_idx += 1
    ###print("====================ダウンサンプリング開始========================")
    ####print("141行 : hの初期値の形は", h.shape)
    #####print("num_resolutionの数は", self.num_resolutions)
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        ####print("\n\n\n\n======================ブロック開始=========================")
        ###print(147, m_idx, modules[m_idx])
        ####print("hs:",hs[-1].shape, "temb:", temb.shape)
        h = modules[m_idx](hs[-1], temb)
        ####print("149行 : hの形は", h.shape)
        m_idx += 1
        ####print("self.attn_resolutions", self.attn_resolutions)
        if h.shape[-1] in self.attn_resolutions:
          ####print("152行 : attn_resolutions")
          h = modules[m_idx](h)
          ####print(156, m_idx, modules[m_idx])
          ####print("157行目 : hの形は", h.shape)
          m_idx += 1
        
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        ####print("num_resolution")
        ####print(163, m_idx, modules[m_idx])
        ####print("164行目 : hの形は", h.shape)
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1
    
    ###print("\n\n\n=============================ダウンサンプリング終了========================\n\n\n")
    ###print("167行 : ダウンサンプリング終了後のhの形は", h.shape)
    h = hs[-1]
    ###print("169行 : hの形は", h.shape)
    ###print(modules[m_idx])
    h = modules[m_idx](h, temb)
    ###print("172行 : hの形は", h.shape)
    m_idx += 1
    ###print(modules[m_idx])
    h = modules[m_idx](h)
    ###print("176行 : hの形は", h.shape)
    m_idx += 1
    ###print(modules[m_idx])
    h = modules[m_idx](h, temb)
    ###print("180行 : hの形は", h.shape)
    m_idx += 1

    # Upsampling block
    ###print("\n\n\n====================アップサンプリング開始===================\n\n\n")
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        ###print("\n\n\n\n=====================ブロック開始====================")
        ###print('188行 : ',i_level,i_block)
        ###print("189行 : hの形は", h.shape)
        ###print("190行 : hsの形は", hs[-1].shape)
        ###print("191行 : catは", torch.cat([h, hs[-1]], dim=1).shape)
        ###print(m_idx, modules[m_idx])
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        ###print("194行 : hの形は", h.shape)
        m_idx += 1
#      if h.shape[-2] in self.attn_resolutions:  #  use y dim
      if h.shape[-1] in self.attn_resolutions:  #  use x dim
        ###print("198行 : attn_resolutions入りまーす")
        ###print(m_idx, modules[m_idx])
        h = modules[m_idx](h)
        ###print("201行 : hの形は", h.shape)
        m_idx += 1
      if i_level != 0:
        ###print("204行 : 入ります")
        ###print(modules[m_idx])
        h = modules[m_idx](h)
        ###print("207行 : hの形は", h.shape)
        m_idx += 1
    ###print("\n\n\n====================アップサンプリング終了！===================\n\n\n")

    ###print("211行 : アップサンプリング終了後のhの形は", h.shape)
    assert not hs
    ####print(modules[m_idx])
    h = self.act(modules[m_idx](h))
    ####print("207行 : hの形は", h.shape)
    m_idx += 1
    ####print(modules[m_idx])
    h = modules[m_idx](h)
    ####print("207行 : hの形は", h.shape)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas
    ###print("229行 : 最終のhの形は", h.shape)
    return h

"""