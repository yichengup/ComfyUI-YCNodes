import torch
import numpy as np
from PIL import Image

# 混合模式列表
BLEND_MODES = [
    'normal', 'multiply', 'screen', 'overlay', 'soft_light', 'hard_light',
    'darken', 'lighten', 'color_dodge', 'color_burn', 'difference',
    'exclusion', 'hue', 'saturation', 'color', 'luminosity'
]

# resize模式
RESIZE_MODES = [
    'contain',  # 等比缩放包含
    'cover'     # 等比缩放覆盖
]

# 对齐方式
ALIGN_MODES = [
    'center',
    'top_left', 'top_center', 'top_right',
    'middle_left', 'middle_right',
    'bottom_left', 'bottom_center', 'bottom_right'
]

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def blend_images(bg, fg, mode='normal', opacity=100):
    """图像混合核心函数"""
    if mode not in BLEND_MODES:
        return fg
    
    bg = np.array(bg).astype(float)
    fg = np.array(fg).astype(float)
    
    # 应用不透明度
    opacity = opacity / 100.0
    
    if mode == 'normal':
        result = fg
    elif mode == 'multiply':
        result = bg * fg / 255.0
    elif mode == 'screen':
        result = 255.0 - ((255.0 - bg) * (255.0 - fg) / 255.0)
    elif mode == 'overlay':
        mask = bg >= 128
        result = np.zeros_like(bg)
        result[mask] = 255.0 - ((255.0 - 2*(bg[mask]-128)) * (255.0 - fg[mask]) / 255.0)
        result[~mask] = (2*bg[~mask] * fg[~mask]) / 255.0
    else:
        # 其他模式默认使用normal
        result = fg
    
    # 应用透明度
    result = bg * (1 - opacity) + result * opacity
    return np.clip(result, 0, 255).astype(np.uint8)

def resize_and_position(image, target_size, resize_mode='contain', x_pos=50, y_pos=50, rotation=0, scale=100, bg_color=(0,0,0)):
    """调整图像大小、位置和旋转"""
    # 第一步：根据resize_mode调整大小
    src_ratio = image.size[0] / image.size[1]
    target_ratio = target_size[0] / target_size[1]
    
    if resize_mode == 'contain':
        if src_ratio > target_ratio:
            new_size = (target_size[0], int(target_size[0] / src_ratio))
        else:
            new_size = (int(target_size[1] * src_ratio), target_size[1])
    else:  # cover
        if src_ratio > target_ratio:
            new_size = (int(target_size[1] * src_ratio), target_size[1])
        else:
            new_size = (target_size[0], int(target_size[0] / src_ratio))
    
    # 第二步：应用scale缩放
    scale_factor = scale / 100.0
    final_size = (int(new_size[0] * scale_factor), int(new_size[1] * scale_factor))
            
    # 第三步：调整图像大小
    resized = image.resize(final_size, Image.LANCZOS)
    
    # 第四步：处理旋转
    if rotation != 0:
        resized = resized.rotate(rotation, Image.BICUBIC, expand=True)
        final_size = resized.size
    
    # 第五步：创建目标画布
    result = Image.new('RGB', target_size, bg_color)
    
    # 第六步：计算最终位置
    # 计算可用空间
    available_width = target_size[0] - final_size[0]
    available_height = target_size[1] - final_size[1]
    
    # 直接使用百分比计算位置
    x = int(available_width * (x_pos / 100))
    y = int(available_height * (y_pos / 100))
    
    # 确保坐标不会超出画布范围
    x = max(0, min(x, available_width))
    y = max(0, min(y, available_height))
    
    # 特殊处理边缘情况
    if y_pos <= 1:  # 顶部对齐
        y = 0
    elif y_pos >= 99:  # 底部对齐
        y = available_height
        
    if x_pos <= 1:  # 左对齐
        x = 0
    elif x_pos >= 99:  # 右对齐
        x = available_width
    
    # 最后：粘贴图像
    result.paste(resized, (x, y))
    return result, (x, y, final_size[0], final_size[1])

class ImageBlendResize:
    def __init__(self):
        self.NODE_NAME = 'ImageBlendResize'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "blend_mode": (BLEND_MODES,),
                "resize_mode": (RESIZE_MODES,),
                "scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 400.0, "step": 0.1}),
                "x_pos": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "y_pos": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_resize"
    CATEGORY = "YiCheng/Image"

    def blend_resize(self, background_image, layer_image, blend_mode, resize_mode, 
                    scale, x_pos, y_pos, rotation, opacity, invert_mask, layer_mask=None):
        # 处理batch
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        
        # 处理背景图batch
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
            
        # 处理前景图batch
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
                
        # 处理mask
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
                
        # 获取最大batch数
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        
        # 处理每个batch
        for i in range(max_batch):
            # 获取当前batch的图像
            bg_tensor = b_images[i] if i < len(b_images) else b_images[-1]
            fg_tensor = l_images[i] if i < len(l_images) else l_images[-1]
            curr_mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            
            # 转换为PIL
            bg_pil = tensor2pil(bg_tensor)
            fg_pil = tensor2pil(fg_tensor)
            
            # 调整前景图大小、位置和旋转
            fg_resized, (x, y, w, h) = resize_and_position(
                fg_pil, bg_pil.size, resize_mode, x_pos, y_pos, rotation, scale
            )
            
            # 同步处理mask
            curr_mask = curr_mask.resize(fg_pil.size, Image.LANCZOS)
            if rotation != 0:
                curr_mask = curr_mask.rotate(rotation, Image.BICUBIC, expand=True)
            curr_mask = curr_mask.resize((w, h), Image.LANCZOS)
            
            # 创建完整尺寸的mask
            full_mask = Image.new('L', bg_pil.size, 0)
            full_mask.paste(curr_mask, (x, y))
            
            # 混合图像
            result_array = blend_images(bg_pil, fg_resized, blend_mode, opacity)
            result_pil = Image.fromarray(result_array)
            
            # 应用mask
            bg_pil.paste(result_pil, (0, 0), mask=full_mask)
            
            # 转换回tensor并添加到结果列表
            ret_images.append(pil2tensor(bg_pil))
        
        # 返回结果
        return (torch.cat(ret_images, dim=0),)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageBlendResize": ImageBlendResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendResize": "Image Blend Resize"
} 