import torch
import numpy as np

class ImageSelector:
    """
    一个用于从多个输入图像中选择指定图像的节点
    支持10个固定输入端口，通过名称选择需要输出的图像
    支持直接输入名称或通过输入端连接
    """
    
    def __init__(self):
        self._valid_names = []  # 缓存有效的名称列表
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "selected_name": ("STRING", {"default": "image1"}),  # 选择的图像名称，支持直接输入或输入端连接
                "image1": ("IMAGE",),
                "name1": ("STRING", {"default": "image1"}),
                "image2": ("IMAGE",),
                "name2": ("STRING", {"default": "image2"}),
                "image3": ("IMAGE",),
                "name3": ("STRING", {"default": "image3"}),
                "image4": ("IMAGE",),
                "name4": ("STRING", {"default": "image4"}),
                "image5": ("IMAGE",),
                "name5": ("STRING", {"default": "image5"}),
                "image6": ("IMAGE",),
                "name6": ("STRING", {"default": "image6"}),
                "image7": ("IMAGE",),
                "name7": ("STRING", {"default": "image7"}),
                "image8": ("IMAGE",),
                "name8": ("STRING", {"default": "image8"}),
                "image9": ("IMAGE",),
                "name9": ("STRING", {"default": "image9"}),
                "image10": ("IMAGE",),
                "name10": ("STRING", {"default": "image10"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_image"
    CATEGORY = "YCNode/Image"
    
    def select_image(self, **kwargs) -> tuple:
        try:
            # 创建图像和名称的映射
            images = {}
            self._valid_names = []
            
            # 收集所有有效的图像和名称
            for i in range(1, 11):
                image_key = f"image{i}"
                name_key = f"name{i}"
                if image_key in kwargs and name_key in kwargs and kwargs[image_key] is not None:
                    name = kwargs[name_key].strip()
                    images[name] = kwargs[image_key]
                    self._valid_names.append(name)
            
            # 检查是否有有效的图像输入
            if not images:
                raise ValueError("没有有效的图像输入")
            
            # 获取selected_name，如果没有提供则使用第一个有效名称
            selected_name = kwargs.get("selected_name")
            if selected_name is None or not selected_name.strip():
                selected_name = self._valid_names[0]
            else:
                selected_name = selected_name.strip()
            
            # 检查选择的名称是否存在
            if selected_name not in images:
                raise ValueError(f"未找到名称: {selected_name}")
            
            # 返回选定的图像
            selected_image = images[selected_name]
            if len(selected_image.shape) == 3:  # 如果是单张图片，增加batch维度
                selected_image = selected_image.unsqueeze(0)
            return (selected_image,)
            
        except Exception as e:
            raise ValueError(f"图像选择失败: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # 总是更新

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 收集有效的名称
        valid_names = []
        for i in range(1, 11):
            name_key = f"name{i}"
            if name_key in kwargs:
                valid_names.append(kwargs[name_key].strip())
        
        # 如果没有有效的图像输入
        if not valid_names:
            return "至少需要一个有效的图像输入"
        
        # 如果提供了selected_name，验证其有效性
        if "selected_name" in kwargs and kwargs["selected_name"]:
            selected_name = kwargs["selected_name"].strip()
            if selected_name and selected_name not in valid_names:
                return f"选择的名称 '{selected_name}' 不在有效名称列表中"
        
        return True

class ImageBatchSelector:
    """
    一个用于图像组合批次处理的节点
    接收批量图像输入，通过自定义名称选择输出指定图像
    支持直接输入名称或通过输入端连接
    """
    
    def __init__(self):
        self._name_list = []  # 缓存名称列表
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 接收批量图像输入
                "names": ("STRING", {"multiline": True, "default": "image1,image2,image3"}),  # 图像名称列表，用逗号分隔
            },
            "optional": {
                "selected_name": ("STRING", {"default": "image1"})  # 选择输出的图像名称，支持直接输入或输入端连接
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_image"
    CATEGORY = "YCNode/Image"
    
    def select_image(self, images: torch.Tensor, names: str, selected_name: str = None) -> tuple:
        try:
            # 处理名称列表
            self._name_list = [name.strip() for name in names.split(",") if name.strip()]
            
            # 检查输入
            if len(self._name_list) == 0:
                raise ValueError("名称列表不能为空")
                
            if len(self._name_list) != images.shape[0]:
                raise ValueError(f"图像数量({images.shape[0]})与名称数量({len(self._name_list)})不匹配")
            
            # 如果没有提供selected_name，使用第一个名称
            if selected_name is None or not selected_name.strip():
                selected_name = self._name_list[0]
            
            # 创建图像和名称的映射
            image_dict = {name: images[i] for i, name in enumerate(self._name_list)}
            
            # 检查选择的名称是否存在
            if selected_name.strip() not in image_dict:
                raise ValueError(f"未找到名称: {selected_name}")
            
            # 返回选定的图像
            selected_image = image_dict[selected_name.strip()]
            if len(selected_image.shape) == 3:  # 如果是单张图片，增加batch维度
                selected_image = selected_image.unsqueeze(0)
            return (selected_image,)
            
        except Exception as e:
            raise ValueError(f"图像选择失败: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # 总是更新

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 首先验证names参数
        if not kwargs["names"].strip():
            return "名称列表不能为空"
        
        # 如果提供了selected_name，验证其有效性
        if "selected_name" in kwargs and kwargs["selected_name"]:
            name_list = [name.strip() for name in kwargs["names"].split(",") if name.strip()]
            selected_name = kwargs["selected_name"].strip()
            if selected_name and selected_name not in name_list:
                return f"选择的名称 '{selected_name}' 不在名称列表中"
        
        return True 

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "ImageSelector": ImageSelector,
    "ImageBatchSelector": ImageBatchSelector
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSelector": "Image Selector",
    "ImageBatchSelector": "Image Batch Selector"
} 