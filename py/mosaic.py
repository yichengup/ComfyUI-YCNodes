import torch
import numpy as np
from PIL import Image

class ImageMosaic:
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

NODE_CLASS_MAPPINGS = {
    "ImageMosaic": ImageMosaic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMosaic": "Image Mosaic"
} 
