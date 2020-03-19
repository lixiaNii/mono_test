############################################
# load images from MVS-Syn, ref mono_dataset
############################################

import os
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image
import imageio
# import PIL.Image as pil
import json
from numpy import linalg as la
# from skimage.transform import rescale
import cv2


from options import nl
from utils import save_depths


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MVSSynDataset(data.Dataset):
    """Superclass for MVS-Syn dataset"""

    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales,
                 is_train=False, img_ext='.jpg'):
        super(MVSSynDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # skip augmentations
        # ...

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def check_depth(self):
        pass

    def get_path(self, folder, frame_index, type='', ext=''):
        path = os.path.join(
            self.data_path,
            "{:04d}".format(int(folder)),
            type,
            "{:04d}{}".format(frame_index, ext))
        return path

    def get_depth(self, scene_index, frame_index, do_flip=False):
        "Load gt depth based on side"
        depth = imageio.imread(self.get_path(scene_index, frame_index, type="depths", ext=".exr"))
        return depth

    def get_color(self, scene_index, frame_index, do_flip=False):
        "Load image based on side"
        color = self.loader(self.get_path(scene_index, frame_index, type="images", ext=".png"))

        # skip do_flip
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def get_cam(self, scene_index, frame_index):
        """Load intrinsics and extrinsics for frame
            intrinsics: unit: pixel
            extrinsics: loaded as world2cam, transform to cam2world and return"""
        filename = self.get_path(scene_index, frame_index, type="poses", ext=".json")
        with open(filename) as f:
            info = json.load(f)
            cx = info['c_x']
            cy = info['c_y']
            fx = info['f_x']
            fy = info['f_y']
            full_intrinsic = np.array([[fx, 0., cx, 0.],
                                       [0., fy, cy, 0],
                                       [0., 0, 1, 0],
                                       [0, 0, 0, 1]])
        return {'extrinsic': la.inv(np.array(info['extrinsic']).astype(np.float32)).squeeze(),
                'intrinsic': full_intrinsic.astype(np.float32)}

    def scale_intrinsics(self, K, full_size=(1280, 720)):
        """Scale intrinsics based on scale and crop
            scale_f: scale factor between original input image and max scale input to network,
                  note: fx, fy: only related to scale; cx, cy: related to both scale and crop"""
        tK = K.copy()
        # cuz may crop height, but scale factor on width is accurate
        scale_f = self.width / full_size[0]
        tK[0, 0] *= scale_f  # fx
        tK[1, 1] *= scale_f  # fy
        tK[0, 2] = tK[0, 2] / full_size[0] * self.width  # cx
        tK[1, 2] = tK[1, 2] / full_size[1] * self.height  # cy
        return tK

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        # crop, cuz mvs-syn: 1280*720 --> 640*352, pil image: (1280, 720)
        bd = 8
        w, h = inputs[("color", 0, -1)].size
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                # inputs[(n, im, i)] = inputs[(n, im, i)][bd:-bd, bd:-bd, :]
                inputs[(n, im, i)] = inputs[(n, im, i)].crop((0, bd, w, h - bd))
            if "gt_depth" in k:
                n, im, i = k
                inputs[(n, im, i)] = inputs[(n, im, i)][bd:-bd, :]
                # save_depths(depths=[inputs[(n, im, i)]], names=["cropped_depth{:02d}".format(im)])

        # scale
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "gt_depth" in k:  # seems no need to re-scale, cuz
                n, im, i = k
                for i in range(self.num_scales):
                    w = self.width // 2 ** i
                    h = self.height // 2 ** i
                    inputs[(n, im, i)] = cv2.resize(inputs[(n, im, -1)], (w, h), interpolation=cv2.INTER_AREA)
                    # inputs[(n, im, i)] = rescale(inputs[(n, im, i - 1)], 1/2)
                    # save_depths(depths=[inputs[(n, im, i)]], names=["scaled_depth{:02d}_s{:02d}".format(im, i)])
                    # inputs[(n, im, i)] = inputs[(n, im, i - 1)]

        # convert to tensor
        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "gt_depth" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return 120 * 30 * 10  # lengthen epochs
        # return len(self.filenames)

    # reference mono_dataset
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # do_color_aug = self.is_train and random.random() > 0.5
        # do_flip = self.is_train and random.random() > 0.5

        # kitti only, cuz from splits/
        # line = self.filenames[index].split()
        # folder = line[0]
        # ...

        # MVS-Syn, 120 scenes, each w/ 100 images
        if nl.fix_sequence:
            scene_index = 0
            frame_index = 10  # frame idx = 10, depth w/ error?
        else:
            scene_index = np.random.randint(0, 10)
            frame_index = np.random.randint(1, 20)

        # load images, poses, gt_depths
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(scene_index, frame_index + i)
            inputs[("gt_depth", i, -1)] = self.get_depth(scene_index, frame_index + i)

            pose = self.get_cam(scene_index, frame_index + i)
            inputs[("intrinsic", i, -1)] = pose["intrinsic"]
            inputs[("extrinsic", i, -1)] = pose["extrinsic"]

        # compute cam_T_cam for source frames, [0,-1,1]
        # seq: w2c * c2w * cam_points
        for i in self.frame_idxs[1:]:
            c2w = inputs[("extrinsic", 0, -1)]  # ref cam
            w2c = la.inv(inputs[("extrinsic", i, -1)])  # src cam
            # inputs[("cam_T_cam", 0, i)] = np.matmul(c2w, w2c)
            inputs[("cam_T_cam", 0, i)] = np.matmul(w2c, c2w)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # K[0, :] *= self.width // (2 ** scale)
            # K[1, :] *= self.height // (2 ** scale)
            # K = K / (2 ** scale)    # cuz K may not be divisible by 2

            # original scale
            K = self.scale_intrinsics(K=inputs[("intrinsic", 0, -1)])
            K[:2, :] = K[:2, :] / (2 ** scale)  # value of other dims should not be changed
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # skip augmentation...
        # if do_color_aug:
        #     color_aug = transforms.ColorJitter.get_params(
        #         self.brightness, self.contrast, self.saturation, self.hue)
        # else:
        #     color_aug = (lambda x: x)
        color_aug = (lambda x: x)

        # resize
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("gt_depth", i, -1)]

        # skip load depth...
        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, side, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs
