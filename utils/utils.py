import os
import shutil
from easydict import EasyDict
import yaml
from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os
import torch
import torch.nn.functional as F


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def mk_path(path, remove=False):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if remove:
                shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(e)


def loadyaml(file_path):
    if file_path is not None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        except IOError as e:
            print(e)
    else:
        print("文件路径为空")
        return None


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


class BoxMaskGenerator(object):
    """
    用于生成多样化的矩形掩码的工具类，适用于图像处理和数据增强任务。
    
    参数：
    - prop_range: tuple[float, float]，控制每个掩码区域面积比例的范围。
    - n_boxes: int，每个样本中生成的矩形区域数量。
    - random_aspect_ratio: bool，是否随机调整矩形区域的长宽比。
    - prop_by_area: bool，是否根据总面积来决定单个区域的比例。
    - within_bounds: bool，是否确保生成的区域在图像边界内。
    - invert: bool，是否反转掩码（即保留背景或前景）。
    """
    def __init__(self,
                 prop_range,
                 n_boxes=1,
                 random_aspect_ratio=True,
                 prop_by_area=True,
                 within_bounds=True,
                 invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array

        生成用于创建掩码的参数。
        
        参数：
        - n_masks: int，需要生成的掩码数量（通常对应批量大小）。
        - mask_shape: tuple[int, int]，掩码的形状（高度，宽度）。
        - rng: [可选] np.random.RandomState 实例，用于控制随机性。
        
        返回：
        - masks: numpy.ndarray，形状为 (n_masks, 1, H, W) 的掩码数组。

        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # 选择应高于阈值的每个掩码的比例
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0
            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        # 是否包含变黄
        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        """
        将生成的掩码参数转换为PyTorch张量。
        
        参数：
        - params: numpy.ndarray，由generate_params方法生成的掩码数组。
        - device: str，指定目标设备（如'cuda'或'cpu'）。
        
        返回：
        - torch.Tensor，形状为 (B, 1, H, W) 的掩码张量。
        """
        return t_params
