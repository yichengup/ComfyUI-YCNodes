import torch
import numpy as np

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

NODE_CLASS_MAPPINGS = {
    "ImageMirror": ImageMirror
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMirror": "Image Mirror"
} 
