import torch
import numpy as np
from PIL import Image

class ImageMirror:
    """图像镜像节点 - 实现整体图像的水平或垂直镜像"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mirror_type": (["horizontal", "vertical"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mirror_image"
    CATEGORY = "YCNode/Image"

    def mirror_image(self, image, mirror_type):
        # 确保输入是正确的格式
        if not isinstance(image, torch.Tensor):
            raise ValueError("输入必须是torch.Tensor类型")
        
        # 转换为numpy数组进行操作
        x = image.cpu().numpy()
        
        if mirror_type == "horizontal":
            # 水平镜像（左右翻转）
            x = np.flip(x, axis=2)
        else:
            # 垂直镜像（上下翻转）
            x = np.flip(x, axis=1)
            
        # 转回tensor并保持设备一致
        result = torch.from_numpy(x.copy()).to(image.device)
        return (result,)


class ImageRotate:
    """图像旋转节点 - 实现90度、180度、270度旋转"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": (["90", "180", "270"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_image"
    CATEGORY = "YCNode/Image"

    def rotate_image(self, image, angle):
        # 确保输入是正确的格式
        if not isinstance(image, torch.Tensor):
            raise ValueError("输入必须是torch.Tensor类型")
        
        # 直接根据角度旋转
        if angle == "90":
            # 顺时针旋转90度
            result = torch.rot90(image, k=3, dims=[1, 2])
        elif angle == "180":
            # 旋转180度
            result = torch.rot90(image, k=2, dims=[1, 2])
        else:  # 270度
            # 顺时针旋转270度
            result = torch.rot90(image, k=1, dims=[1, 2])
            
        return (result,)


class ImageMosaic:
    """图像马赛克节点 - 实现图像马赛克效果"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mosaic_size": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mosaic"
    CATEGORY = "YCNode/Image"

    def apply_mosaic(self, image, mosaic_size):
        # 将图像转换为numpy数组
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # 确保图像在0-1范围内
        if image.max() > 1.0:
            image = image / 255.0
            
        # 获取图像尺寸
        h, w = image.shape[1:3]
        
        # 计算每个马赛克块的尺寸
        block_h = h // mosaic_size
        block_w = w // mosaic_size
        
        # 创建新图像
        mosaic_image = image.copy()
        
        # 应用马赛克效果
        for i in range(0, h - block_h + 1, block_h):
            for j in range(0, w - block_w + 1, block_w):
                # 计算块的平均颜色
                block = image[:, i:i+block_h, j:j+block_w]
                mean_color = np.mean(block, axis=(1, 2), keepdims=True)
                
                # 将块填充为平均颜色
                mosaic_image[:, i:i+block_h, j:j+block_w] = mean_color
        
        # 转换回torch tensor
        return (torch.from_numpy(mosaic_image),)


# 注册所有节点
NODE_CLASS_MAPPINGS = {
    "ImageMirror": ImageMirror,
    "ImageRotate": ImageRotate,
    "ImageMosaic": ImageMosaic
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMirror": "Image Mirror",
    "ImageRotate": "Image Rotate", 
    "ImageMosaic": "Image Mosaic"
} 
