# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
# NL::
# data_dir = "/mnt/win_data2/data/lixia"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# NL::params
# ===========================================
class NLParams():
    def __init__(self):

        # set server
        self.server = "zju"  # cai4, haiyong
        self.data_dir = {"zju": "/data/lixia/",
                         "cai4": "/mnt/win_data2/data/lixia/"}
        self.data_dir = self.data_dir[self.server]

        # set dataset
        self.dataset = "mvs_syn"  # ["kitti", "kitti_odom", "kitti_depth", "kitti_test"]
        # self.use_mvs = True  # decide which dataset to use
        # self.use_odom = False  # kitti odom
        # self.use_kitti = False  # kitti raw
        self.num_batch = 1  # orig: 12
        self.num_workers = self.num_batch  # orig: 12

        # Train setting
        self.use_refine = True  # same basic structure w/ original decoder
        self.use_gt_pose = True  # use gt pose to transform if True
        self.use_gt_depth = False
        self.fix_sequence = False  # train on single sequence if True

        self.load_pretrain = {"encoder": True, "depth": True,
                              "pose_encoder": True, "pose": True}

        self.skip_valid = True
        self.skip_aug = True  # skip data augmentation
        self.learning_rate = 5e-5  # original: 1e-4

        # RefDepthDecoder settings
        self.settings = {"use_warp": False, "use_coarse": False,
                         "use_orig": False, "use_source": False,
                         "out_residual": False}

        # Dataset settings
        self.kitti = {"data_path": os.path.join(self.data_dir, "KITTI/kitti_data"),
                      "eval_split": "eigen",
                      "model_name": "mono+stereo_640x192"}
        self.odom = {"data_path": os.path.join(self.data_dir, "KITTI/odom"),
                     "eval_split": "odom_9",
                     "model_name": "mono+stereo_odom_640x192"}
        self.mvs = {"data_path": os.path.join(self.data_dir, "MVS_Syn/GTAV_720"),
                    "eval_split": "eigen",
                    "model_name": "mono+stereo_640x192"}  # use kitti trained model

        if self.dataset == "mvs_syn":
            self.data_settings = self.mvs
        elif self.dataset == "kitti_odom":
            self.data_settings = self.odom
        elif self.dataset == "kitti":
            self.data_settings = self.kitti

        # Path settings
        self.log_dir = os.path.join(self.data_dir, "_rst/mono_test/logs")
        self.cache_root = os.path.join(self.data_dir, "_rst/mono_test/cache")

        self.well_trained = True
        if self.well_trained:
            # load well trained models from monodepth2
            self.model_path = os.path.join("models", self.data_settings["model_name"])  # if not load, set None
        else:
            # load testing models
            model_name = "03-26_23-08-41_mdp/models/weights_331"
            self.model_path = os.path.join(self.log_dir, model_name)
        # self.log_dir = "/mnt/win_data2/data/lixia/_rst/mono_test/logs"
        # self.cache_root = "/mnt/win_data2/data/lixia/_rst/mono_test/cache"


nl = NLParams()
# print out params
nl_dict = nl.__dict__
for key in nl_dict:
    print(key, ":", nl_dict[key])

# ===========================================

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # for run on jupyter, not sure why
        self.parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=nl.data_settings["data_path"])
        # default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=nl.log_dir)
        # default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="odom")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default=nl.dataset,  # "kitti_odom"
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=352 if nl.dataset == "mvs_syn" else 192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)  # 640
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=6000.0)  # originally 100.
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=nl.num_batch)  # NL::default is 12
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=nl.learning_rate)  # 1e-4
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20 * 120)  # 20
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=nl.num_workers)  # 12

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default=nl.model_path if nl.load_pretrain else None)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=50)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default=nl.data_settings["eval_split"],  # "eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
