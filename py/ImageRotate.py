import torch

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

NODE_CLASS_MAPPINGS = {
    "ImageRotate": ImageRotate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRotate": "Image Rotate"
} 
