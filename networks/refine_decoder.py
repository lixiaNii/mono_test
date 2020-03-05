##################################
# test my implementation
##################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, enc_channels, source_num=1, projection=None, dconverter=None,
                 scale_num=4, use_warp=False, use_skips=True, use_coarse=False, use_orig=False):
        super().__init__()

        self.projection = projection
        self.dconverter = dconverter
        self.source_num = source_num
        self.scale_num = scale_num
        self.scales = range(self.scale_num)
        self.use_skips = use_skips
        # control differences between mine and mono decoder ++++++++++++++++++++++
        self.use_warp = use_warp  # whether use coarse depth
        self.use_coarse = use_coarse  # whether input coarse depth
        self.use_orig = True if self.use_warp else use_orig  # TODO: note
        # whether use original image,
        # cuz monodepth2 only use resnet features
        # Note: if use warp, must use orig
        # but use orig may not use warp
        self.coarse_num = 3  # number of coarse depth
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.num_ch_enc = enc_channels  # scale down sequence
        if self.use_orig:
            self.num_ch_enc = np.insert(self.encoder.num_ch_enc, 0, 3)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # most same w/ monodepth2, scale down sequence

        self.convs = OrderedDict()

        for i in range(4, -1, -1):
            # upconv_0, same w/ mono decoder
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1, after upsample, add skipped/warped/coarse feature if exist
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips:
                t = i if self.use_orig else i - 1
                if t > 0:
                    num_ch_in += self.num_ch_enc[t] * self.source_num
                    # input warped features, must use orig
                    if self.use_warp and t < self.coarse_num:
                        num_ch_in += self.num_ch_enc[t] * self.source_num
                    # input coarse depth
                    if self.use_coarse and i < self.coarse_num:
                        num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            # original decoder codes
            # if self.use_skips:
            #     if self.use_warp:
            #         num_ch_in += self.num_ch_enc[i] * self.source_num if i > 2 \
            #             else self.num_ch_enc[i] * self.source_num * 2 + 1   # warped feature and coarse depth
            #     else:
            #         if self.use_coarse:
            #             num_ch_in += self.num_ch_enc[i] * self.source_num if i > 2 \
            #                 else self.num_ch_enc[i] * self.source_num + 1
            #         else:
            #             num_ch_in += self.num_ch_enc[i] * self.source_num
            # num_ch_out = self.num_ch_dec[i]
            # self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("depthconv", s)] = Conv3x3(self.num_ch_dec[s], 1)  # output depth residual
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def warp_block(self, depth, feats, poses):
        """Warp features based on given depth and poses
            depth: depth on target view
            poses: target pose @ 0
            feats: source features
            return: warped source features"""
        depth = self.dconverter.to_real_depth(depth)
        feats_warped = [self.projection.backwarp(
            tar={'depth': depth, 'pose': poses[0]},
            src={'rgb': f, 'pose': p})[0]
                        for f, p in zip(feats, poses[1:])]
        return feats_warped

    def forward(self, input_features, input_depth=None, poses=None):
        self.outputs = {}

        # reverse scale
        if input_depth is not None:
            input_depth.reverse()

        # decoder
        x = input_features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            t = i if self.use_orig else i - 1  # index shift if use original image
            if self.use_skips:
                if t > 0:
                    x += input_features[t]
                    if self.use_warp and t < self.coarse_num:
                        feats_warped = self.warp_block(depth=input_depth[t],
                                                       feats=input_features[t],
                                                       poses=poses)
                        x += feats_warped
                    if self.use_coarse and i < self.coarse_num:
                        x += [input_depth[t]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)

            # output
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("depthconv", 2)](x)) - 0.5
                if self.out_residual:
                    self.outputs[("disp", i)] += input_depth[t]

        return self.outputs

    #
    # def tforward(self, input_features, input_depth, poses):
    #     """Instead of input refined depth, but input feature for decoded to disparity,
    #         modified from monodepth2: use simple layers first
    #         Note: output by cur scale: residual for cur scale,
    #               input features are w/ original image"""
    #
    #     self.outputs = {}
    #
    #     # 8
    #     feat5 = torch.cat(input_features[5], dim=1)
    #     feat5 = self.convs[("upconv", 4, 0)](feat5)
    #     # up_feat5 = self.deconv[4](feat5)
    #     up_feat5 = upsample(feat5)
    #
    #     # 16
    #     feat4 = torch.cat([up_feat5] + input_features[4], dim=1)
    #     feat4 = self.convs[("upconv", 4, 1)](feat4)
    #     feat4 = self.convs[("upconv", 3, 0)](feat4)
    #     # up_feat4 = self.deconv[3](feat4)
    #     up_feat4 = upsample(feat4)
    #
    #     # 32
    #     feat3 = torch.cat([up_feat4] + input_features[3], dim=1)
    #     feat3 = self.convs[("upconv", 3, 1)](feat3)
    #     feat3 = self.convs[("upconv", 2, 0)](feat3)
    #     # up_feat3 = self.deconv[2](feat3)
    #     up_feat3 = upsample(feat3)
    #
    #     # 64
    #     if self.use_warp:
    #         feat2 = [input_depth[0]] + [up_feat3] + input_features[2]
    #         input_features_warped2 = self.warp_block(depth=input_depth[0],
    #                                                  feats=input_features[2],
    #                                                  poses=poses)
    #         # TODO: if more than 1 source, may need to change cat sequence, -> feat0, feat0_warped, feat1...
    #         feat2 = torch.cat(feat2 + input_features_warped2, dim=1)
    #     else:
    #         feat2 = torch.cat([up_feat3] + input_features[2], dim=1)
    #     feat2 = self.convs[("upconv", 2, 1)](feat2)
    #     depth_residual2 = self.sigmoid(self.convs[("depthconv", 2)](feat2)) - 0.5
    #     depth_refined2 = depth_residual2 + input_depth[0]
    #     feat2 = self.convs[("upconv", 1, 0)](feat2)
    #     # up_feat2 = self.deconv[1](feat2)
    #     up_feat2 = upsample(feat2)
    #
    #     # 128
    #     # coarse_depth1 = F.interpolate(depth_refined2, scale_factor=2)
    #     coarse_depth1 = input_depth[1]
    #     if self.use_warp:
    #         feat1 = [coarse_depth1] + [up_feat2] + input_features[1]
    #         input_features_warped1 = self.warp_block(depth=coarse_depth1,
    #                                                  feats=input_features[1],
    #                                                  poses=poses)
    #         feat1 = torch.cat(feat1 + input_features_warped1, dim=1)
    #         # feat1 = torch.cat([up_feat2] + input_features[1] + input_features_warped1, dim=1)
    #     else:
    #         feat1 = torch.cat([up_feat2] + input_features[1], dim=1)
    #     feat1 = self.convs[("upconv", 1, 1)](feat1)
    #     depth_residual1 = self.sigmoid(self.convs[("depthconv", 1)](feat1)) - 0.5
    #     depth_refined1 = depth_residual1 + coarse_depth1
    #     feat1 = self.convs[("upconv", 0, 0)](feat1)
    #     # up_feat1 = self.deconv[0](feat1)
    #     up_feat1 = upsample(feat1)
    #
    #     # 256
    #     # coarse_depth0 = F.interpolate(depth_refined1, scale_factor=2)
    #     coarse_depth0 = input_depth[2]
    #     if self.use_warp:
    #         feat0 = [coarse_depth0] + [up_feat1] + input_features[0]
    #         input_features_warped0 = self.warp_block(depth=coarse_depth0,
    #                                                  feats=input_features[0],
    #                                                  poses=poses)
    #         feat0 = torch.cat(feat0 + input_features_warped0, dim=1)
    #     else:
    #         feat0 = torch.cat([up_feat1] + input_features[0], dim=1)
    #     feat0 = self.convs[("upconv", 0, 1)](feat0)
    #     depth_residual0 = self.sigmoid(self.convs[("depthconv", 0)](feat0)) - 0.5
    #     depth_refined0 = coarse_depth0 + depth_residual0
    #
    #     self.outputs[("disp", 2)] = depth_refined2
    #     self.outputs[("disp", 1)] = depth_refined1
    #     self.outputs[("disp", 0)] = depth_refined0
    #
    #     return self.outputs
    #     # return [depth_refined2, depth_refined1, depth_refined0]
