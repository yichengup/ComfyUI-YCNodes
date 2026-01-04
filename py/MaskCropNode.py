import torch
import numpy as np
import cv2
import math
from PIL import Image

# 填充模式列表
PAD_MODES = [
    'edge',      # 边缘颜色填充
    'reflect',   # 镜像反射填充
    'symmetric', # 对称填充
    'wrap',      # 循环填充
    'constant'   # 固定颜色填充
]

def hex_to_rgb(hex_color):
    """十六进制颜色转RGB元组"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def smart_pad_image(image, left, top, right, bottom, pad_mode='edge', fill_color='#ffffff'):
    """智能图像填充函数"""
    if pad_mode == 'constant':
        # 固定颜色填充
        try:
            rgb_color = hex_to_rgb(fill_color)
        except:
            rgb_color = (255, 255, 255)  # 默认白色
        
        # 根据图像模式选择合适的颜色值
        if image.mode == 'L':
            # 灰度图像：将RGB转换为灰度值 (使用标准公式: 0.299*R + 0.587*G + 0.114*B)
            fill_color_value = int(0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2])
        elif image.mode in ('RGB', 'RGBA'):
            fill_color_value = rgb_color
        else:
            # 其他模式，尝试使用第一个通道的值
            fill_color_value = rgb_color[0] if isinstance(rgb_color, tuple) else rgb_color
        
        # 使用PIL的pad方法
        result = image.copy()
        result = result.crop((0, 0, image.width + left + right, image.height + top + bottom))
        result.paste(image, (left, top))
        
        # 填充边缘
        if left > 0:
            fill_img = Image.new(image.mode, (left, image.height), fill_color_value)
            result.paste(fill_img, (0, top))
        if right > 0:
            fill_img = Image.new(image.mode, (right, image.height), fill_color_value)
            result.paste(fill_img, (left + image.width, top))
        if top > 0:
            fill_img = Image.new(image.mode, (image.width + left + right, top), fill_color_value)
            result.paste(fill_img, (0, 0))
        if bottom > 0:
            fill_img = Image.new(image.mode, (image.width + left + right, bottom), fill_color_value)
            result.paste(fill_img, (0, top + image.height))
            
    else:
        # 使用OpenCV的copyMakeBorder方法
        img_array = np.array(image)
        
        # 处理RGBA图像
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA图像需要特殊处理
            if pad_mode == 'edge':
                border_type = cv2.BORDER_REPLICATE
            elif pad_mode == 'reflect':
                border_type = cv2.BORDER_REFLECT_101
            elif pad_mode == 'symmetric':
                border_type = cv2.BORDER_REFLECT
            elif pad_mode == 'wrap':
                border_type = cv2.BORDER_WRAP
            else:
                border_type = cv2.BORDER_REPLICATE
            
            # 分别处理RGB和Alpha通道
            rgb_channels = img_array[:, :, :3]
            alpha_channel = img_array[:, :, 3]
            
            # 填充RGB通道
            rgb_padded = cv2.copyMakeBorder(
                rgb_channels, top, bottom, left, right, 
                border_type, value=[0, 0, 0]
            )
            
            # 填充Alpha通道
            alpha_padded = cv2.copyMakeBorder(
                alpha_channel, top, bottom, left, right, 
                border_type, value=255
            )
            
            # 合并通道
            result_array = np.dstack((rgb_padded, alpha_padded))
            result = Image.fromarray(result_array)
            
        else:
            # RGB或灰度图像
            if pad_mode == 'edge':
                border_type = cv2.BORDER_REPLICATE
            elif pad_mode == 'reflect':
                border_type = cv2.BORDER_REFLECT_101
            elif pad_mode == 'symmetric':
                border_type = cv2.BORDER_REFLECT
            elif pad_mode == 'wrap':
                border_type = cv2.BORDER_WRAP
            else:
                border_type = cv2.BORDER_REPLICATE
            
            result_array = cv2.copyMakeBorder(
                img_array, top, bottom, left, right, 
                border_type, value=[0, 0, 0]
            )
            result = Image.fromarray(result_array)
    
    return result

# 依赖函数 - 从 imagefunc.py 迁移
def log(message: str, message_type: str = 'info'):
    name = 'YCNode'
    
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;34m' + message + '\033[m'
    
    print(f"[{name}] {message}")

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image2mask(image: Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

class MaskCrop_YC:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_frame": ("MASK",),
                "top_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "bottom_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "left_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "right_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "round_to_multiple": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "detection_method": (["mask_area"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "COORDS")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_frame", "crop_frame_preview", "crop_coords")
    FUNCTION = "crop_with_mask"
    CATEGORY = "YCNode/Mask"

    def crop_with_mask(self, image, crop_frame, top_padding, bottom_padding, left_padding, right_padding, round_to_multiple, invert_mask, detection_method, mask=None):
        # 如果没有提供mask，创建一个全零mask
        if mask is None:
            # 创建一个与图像尺寸相同的全零mask
            mask = torch.zeros((1, image.shape[2], image.shape[3]), device=image.device)
        
        # 确保输入是正确的格式
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(crop_frame, torch.Tensor):
            crop_frame = crop_frame.cpu().numpy()
            
        # 处理mask维度
        if len(mask.shape) == 3:
            mask = mask[0]
        elif len(mask.shape) == 4:
            mask = mask[0, 0]
            
        if len(crop_frame.shape) == 3:
            crop_frame = crop_frame[0]
        elif len(crop_frame.shape) == 4:
            crop_frame = crop_frame[0, 0]

        # 如果需要反转遮罩
        if invert_mask:
            crop_frame = 1 - crop_frame

        # 转换为二值图像
        binary_mask = (crop_frame > 0.5).astype(np.uint8)
        
        # 找到mask的边界框
        if detection_method == "mask_area":
            y_indices, x_indices = np.nonzero(binary_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                # 创建一个空的预览图像，直接使用原始图像
                original_img_np = image[0].cpu().numpy()  # [H, W, C]
                if original_img_np.dtype != np.float32:
                    original_img_np = original_img_np.astype(np.float32)
                if original_img_np.max() > 1.0:
                    original_img_np = original_img_np / 255.0
                    
                # 直接使用原始图像，不添加任何标注
                empty_preview_tensor = torch.from_numpy(original_img_np).float()
                empty_preview_tensor = empty_preview_tensor.unsqueeze(0)  # 添加批次维度
                return (image, mask, crop_frame, empty_preview_tensor, (0, 0, image.shape[3], image.shape[2]))
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 添加padding
            x_min = max(0, x_min - left_padding)
            x_max = min(binary_mask.shape[1], x_max + right_padding)
            y_min = max(0, y_min - top_padding)
            y_max = min(binary_mask.shape[0], y_max + bottom_padding)
            
            # 确保尺寸是round_to_multiple的倍数
            width = x_max - x_min
            height = y_max - y_min
            
            new_width = ((width + round_to_multiple - 1) // round_to_multiple) * round_to_multiple
            new_height = ((height + round_to_multiple - 1) // round_to_multiple) * round_to_multiple
            
            # 调整padding以达到所需尺寸
            x_pad = new_width - width
            y_pad = new_height - height
            
            x_min = max(0, x_min - x_pad // 2)
            x_max = min(binary_mask.shape[1], x_max + (x_pad - x_pad // 2))
            y_min = max(0, y_min - y_pad // 2)
            y_max = min(binary_mask.shape[0], y_max + (y_pad - y_pad // 2))
            
            # 裁剪图像和mask
            cropped_image = image[:, y_min:y_max, x_min:x_max, :]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            cropped_crop_frame = crop_frame[y_min:y_max, x_min:x_max]
            
            # 创建预览用的crop_frame - 改为图片格式，并绘制红色边框
            # 首先将图像转换到numpy进行处理
            original_img_np = image[0].cpu().numpy()  # [H, W, C]
            
            # 确保是RGB格式，且值范围为[0,1]
            if original_img_np.dtype != np.float32:
                original_img_np = original_img_np.astype(np.float32)
            if original_img_np.max() > 1.0:
                original_img_np = original_img_np / 255.0
                
            # 创建预览图像副本，不显示半透明遮罩，只显示边框
            preview_img = original_img_np.copy()
            
            # 找到原始mask的边界 - 未添加padding的原始区域
            orig_y_indices, orig_x_indices = np.nonzero(binary_mask)
            orig_x_min, orig_x_max = np.min(orig_x_indices), np.max(orig_x_indices)
            orig_y_min, orig_y_max = np.min(orig_y_indices), np.max(orig_y_indices)
            
            # 确保坐标是整数
            x_min_int, y_min_int = int(x_min), int(y_min)
            x_max_int, y_max_int = int(x_max), int(y_max)
            orig_x_min_int, orig_y_min_int = int(orig_x_min), int(orig_y_min)
            orig_x_max_int, orig_y_max_int = int(orig_x_max), int(orig_y_max)
            
            # 定义颜色
            red_color = np.array([1.0, 0.0, 0.0])    # 红色用于标注最终裁剪区域（含padding）
            green_color = np.array([0.0, 1.0, 0.0])  # 绿色用于标注多出的padding区域
            
            # 线宽
            line_width = 2
            
            # 绘制红色边框（最终裁剪区域，包含padding）
            # 顶部边框
            preview_img[y_min_int:y_min_int+line_width, x_min_int:x_max_int] = red_color
            # 底部边框
            preview_img[y_max_int-line_width:y_max_int, x_min_int:x_max_int] = red_color
            # 左侧边框
            preview_img[y_min_int:y_max_int, x_min_int:x_min_int+line_width] = red_color
            # 右侧边框
            preview_img[y_min_int:y_max_int, x_max_int-line_width:x_max_int] = red_color
            
            # 绘制绿色边框（原始遮罩区域，不含padding）
            # 只有当原始区域与裁剪区域不同时才绘制绿色边框
            if (orig_x_min_int != x_min_int or orig_y_min_int != y_min_int or 
                orig_x_max_int != x_max_int or orig_y_max_int != y_max_int):
                # 顶部边框
                preview_img[orig_y_min_int:orig_y_min_int+line_width, orig_x_min_int:orig_x_max_int] = green_color
                # 底部边框
                preview_img[orig_y_max_int-line_width:orig_y_max_int, orig_x_min_int:orig_x_max_int] = green_color
                # 左侧边框
                preview_img[orig_y_min_int:orig_y_max_int, orig_x_min_int:orig_x_min_int+line_width] = green_color
                # 右侧边框
                preview_img[orig_y_min_int:orig_y_max_int, orig_x_max_int-line_width:orig_x_max_int] = green_color
            
            # 转换回PyTorch张量格式 [B, H, W, C]
            preview_img_tensor = torch.from_numpy(preview_img).float()
            preview_img_tensor = preview_img_tensor.unsqueeze(0)  # 添加批次维度
            
            # 转换回torch tensor
            cropped_mask = torch.from_numpy(cropped_mask).float()
            cropped_crop_frame = torch.from_numpy(cropped_crop_frame).float()
            
            if len(cropped_mask.shape) == 2:
                cropped_mask = cropped_mask.unsqueeze(0)
            if len(cropped_crop_frame.shape) == 2:
                cropped_crop_frame = cropped_crop_frame.unsqueeze(0)
                
            # 返回裁剪坐标 (x_min, y_min, x_max, y_max)
            crop_coords = (int(x_min), int(y_min), int(x_max), int(y_max))
            return (cropped_image, cropped_mask, cropped_crop_frame, preview_img_tensor, crop_coords)
        
        # 如果没有检测到有效区域，返回原始图像和空的预览图
        original_img_np = image[0].cpu().numpy()
        if original_img_np.dtype != np.float32:
            original_img_np = original_img_np.astype(np.float32)
        if original_img_np.max() > 1.0:
            original_img_np = original_img_np / 255.0
        
        # 直接使用原始图像，不添加任何标注
        empty_preview_tensor = torch.from_numpy(original_img_np).float()
        empty_preview_tensor = empty_preview_tensor.unsqueeze(0)  # 添加批次维度
        
        return (image, mask, crop_frame, empty_preview_tensor, (0, 0, image.shape[3], image.shape[2]))

class MaskCropRestore_YC:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "crop_coords": ("COORDS",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cropped_mask": ("MASK",),
                "crop_frame": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "inpaint_mask")
    FUNCTION = "restore_crop"
    CATEGORY = "YCNode/Mask"

    def restore_crop(self, background_image, cropped_image, crop_coords, invert_mask, cropped_mask=None, crop_frame=None):
        x_min, y_min, x_max, y_max = crop_coords
        
        # 正确解析图像维度
        # 图像形状: [batch, height, width, channels]
        batch_size = background_image.shape[0]
        bg_height = background_image.shape[1]
        bg_width = background_image.shape[2]
        channels = background_image.shape[3]
        
        # 检查图像维度是否合理
        if cropped_image.shape[1] < 10 or cropped_image.shape[2] < 10:
            # 修复坐标 - 确保至少有1像素的差距
            if x_min >= x_max:
                x_max = x_min + 1
            if y_min >= y_max:
                y_max = y_min + 1
        
        # 强制确保裁剪区域有合理的高度和宽度（至少10像素）
        if y_max - y_min < 10:
            height_to_add = 10 - (y_max - y_min)
            y_min = max(0, y_min - height_to_add // 2)
            y_max = min(bg_height, y_max + (height_to_add - height_to_add // 2))
            
        if x_max - x_min < 10:
            width_to_add = 10 - (x_max - x_min)
            x_min = max(0, x_min - width_to_add // 2)
            x_max = min(bg_width, x_max + (width_to_add - width_to_add // 2))
        
        # 确保坐标在有效范围内
        x_min = max(0, min(x_min, bg_width - 1))
        y_min = max(0, min(y_min, bg_height - 1))
        x_max = max(x_min + 1, min(x_max, bg_width))
        y_max = max(y_min + 1, min(y_max, bg_height))
        
        # 创建输出图像和mask（与背景图像相同大小）
        output_image = background_image.clone()
        output_mask = torch.zeros((1, bg_height, bg_width), device=background_image.device)
        
        # 确保裁剪图像尺寸与目标区域匹配
        expected_height = y_max - y_min
        expected_width = x_max - x_min
        
        # 验证裁剪图像尺寸
        if cropped_image.shape[1] != expected_height or cropped_image.shape[2] != expected_width:
            # 调整放置策略
            if expected_width == 0 or expected_height == 0:
                # 使用安全的默认值
                x_min, y_min = 0, 0
                x_max = min(cropped_image.shape[2], bg_width)
                y_max = min(cropped_image.shape[1], bg_height)
                expected_width = x_max - x_min
                expected_height = y_max - y_min
            else:
                # 尝试调整坐标以匹配实际图像尺寸
                src_height, src_width = cropped_image.shape[1], cropped_image.shape[2]
                
                # 如果输入图像尺寸小于目标区域，直接使用输入图像尺寸
                if src_height < expected_height:
                    y_max = y_min + src_height
                if src_width < expected_width:
                    x_max = x_min + src_width
                
                # 检查调整后尺寸是否合理
                if y_max - y_min < 10:
                    y_max = min(y_min + 10, bg_height)
                    
                if x_max - x_min < 10:
                    x_max = min(x_min + 10, bg_width)
                
                # 如果输入图像尺寸大于目标区域，可能需要裁剪输入图像
                expected_height = y_max - y_min
                expected_width = x_max - x_min
        
        # 将裁剪的图像放回原位置
        try:
            # 确保区域有效
            if expected_width <= 0 or expected_height <= 0:
                raise ValueError(f"无效的区域大小: 宽度={expected_width}, 高度={expected_height}")
                
            # 安全复制，考虑到输入图像可能与目标区域大小不匹配
            copy_height = min(expected_height, cropped_image.shape[1])
            copy_width = min(expected_width, cropped_image.shape[2])
            
            if copy_width > 0 and copy_height > 0:
                # 先直接复制图像（后续会用遮罩进行混合）
                output_image[:, y_min:y_min+copy_height, x_min:x_min+copy_width, :] = cropped_image[:, :copy_height, :copy_width, :]
                
        except RuntimeError:
            # 如果还是失败，尝试最保守的方法
            try:
                min_height = min(cropped_image.shape[1], bg_height - y_min)
                min_width = min(cropped_image.shape[2], bg_width - x_min)
                
                if min_height > 0 and min_width > 0:
                    output_image[:, y_min:y_min+min_height, x_min:x_min+min_width, :] = cropped_image[:, :min_height, :min_width, :]
                    # 更新复制区域的大小，供后面遮罩使用
                    copy_height, copy_width = min_height, min_width
            except:
                pass
        
        # 安全处理mask
        try:
            # 确定要使用的遮罩
            mask_to_use = None
            if cropped_mask is not None:
                mask_to_use = cropped_mask
            elif crop_frame is not None:
                mask_to_use = crop_frame
            else:
                # 如果两个遮罩都没有提供，创建一个全1遮罩（即选中整个裁剪区域）
                # 使用与裁剪图像一致的尺寸
                height, width = cropped_image.shape[1], cropped_image.shape[2]
                mask_to_use = torch.ones((1, height, width), device=cropped_image.device)
            
            # 将裁剪的遮罩转换为正确的格式
            if isinstance(mask_to_use, torch.Tensor):
                if len(mask_to_use.shape) == 4:
                    mask_to_use = mask_to_use[0]
                
                # 确保mask的维度至少是3维 [C, H, W]
                if len(mask_to_use.shape) == 2:
                    mask_to_use = mask_to_use.unsqueeze(0)
                
                # 验证遮罩维度
                if len(mask_to_use.shape) != 3:
                    return (output_image, output_mask)
                    
                # 验证遮罩尺寸
                mask_height, mask_width = mask_to_use.shape[1], mask_to_use.shape[2]
                
                # 检查遮罩尺寸是否异常小
                if mask_height < 10 or mask_width < 10:
                    # 如果遮罩高度或宽度异常小，尝试调整
                    if expected_height > 10 and expected_width > 10:
                        try:
                            # 创建新的遮罩并调整大小
                            new_mask = torch.nn.functional.interpolate(
                                mask_to_use.unsqueeze(0) if len(mask_to_use.shape) == 3 else mask_to_use,
                                size=(expected_height, expected_width),
                                mode='nearest'
                            )
                            mask_to_use = new_mask.squeeze(0) if len(mask_to_use.shape) == 3 else new_mask
                            mask_height, mask_width = expected_height, expected_width
                        except:
                            pass
                
                # 使用已经调整过的目标区域大小
                copy_height = min(copy_height if 'copy_height' in locals() else expected_height, mask_height)
                copy_width = min(copy_width if 'copy_width' in locals() else expected_width, mask_width)
                
                # 确保copy_height和copy_width至少有10像素（防止线状遮罩）
                copy_height = max(10, copy_height)
                copy_width = max(10, copy_width)
                
                # 确保不超出背景边界
                if y_min + copy_height > bg_height:
                    copy_height = bg_height - y_min
                if x_min + copy_width > bg_width:
                    copy_width = bg_width - x_min
                
                if copy_width > 0 and copy_height > 0:
                    # 确保索引有效
                    if copy_height > mask_to_use.shape[1]:
                        copy_height = mask_to_use.shape[1]
                    if copy_width > mask_to_use.shape[2]:
                        copy_width = mask_to_use.shape[2]
                    
                    # 安全复制遮罩 - 修正索引
                    try:
                        output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = mask_to_use[:, :copy_height, :copy_width]
                    except:
                        try:
                            # 强制重新调整遮罩尺寸
                            adjusted_mask = torch.nn.functional.interpolate(
                                mask_to_use.unsqueeze(0) if len(mask_to_use.shape) == 3 else mask_to_use,
                                size=(copy_height, copy_width),
                                mode='nearest'
                            )
                            adjusted_mask = adjusted_mask.squeeze(0) if len(mask_to_use.shape) == 3 else adjusted_mask
                            
                            # 再次尝试放置
                            output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = adjusted_mask
                        except:
                            pass
                    
                    # 验证输出遮罩
                    nonzero_count = torch.count_nonzero(output_mask)
                    if nonzero_count == 0:
                        # 最后尝试 - 创建一个简单的矩形遮罩
                        try:
                            output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = 1.0
                        except:
                            pass
        except RuntimeError:
            # 如果还是失败，尝试最保守的方法
            try:
                if 'mask_to_use' in locals() and isinstance(mask_to_use, torch.Tensor):
                    mask_height, mask_width = mask_to_use.shape[1], mask_to_use.shape[2]
                    min_height = min(mask_height, bg_height - y_min)
                    min_width = min(mask_width, bg_width - x_min)
                    
                    # 确保最小高度和宽度不小于10像素
                    min_height = max(10, min_height)
                    min_width = max(10, min_width)
                    
                    # 确保不超出背景边界
                    if y_min + min_height > bg_height:
                        min_height = bg_height - y_min
                    if x_min + min_width > bg_width:
                        min_width = bg_width - x_min
                    
                    if min_height > 0 and min_width > 0:
                        # 确保不超出mask边界
                        mask_part = mask_to_use[:, :min(min_height, mask_height), :min(min_width, mask_width)]
                        # 确保目标区域足够大
                        if y_min + mask_part.shape[1] <= bg_height and x_min + mask_part.shape[2] <= bg_width:
                            output_mask[:, y_min:y_min+mask_part.shape[1], x_min:x_min+mask_part.shape[2]] = mask_part
            except:
                pass
                
            # 最后的尝试：创建一个简单的矩形遮罩
            try:
                valid_height = min(10, bg_height - y_min)
                valid_width = min(10, bg_width - x_min)
                if valid_height > 0 and valid_width > 0:
                    output_mask[:, y_min:y_min+valid_height, x_min:x_min+valid_width] = 1.0
            except:
                pass
        
        if invert_mask:
            output_mask = 1 - output_mask
        
        # 使用遮罩进行图像合成
        try:
            # 确保遮罩和图像尺寸匹配
            if output_mask.shape[1] == bg_height and output_mask.shape[2] == bg_width:
                # 获取裁剪区域的遮罩
                mask_region = output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width]
                
                # 将遮罩扩展到RGB通道 [B, H, W, C]
                mask_expanded = mask_region.unsqueeze(-1).expand(-1, -1, -1, channels)
                
                # 获取背景图像的对应区域
                bg_region = background_image[:, y_min:y_min+copy_height, x_min:x_min+copy_width, :]
                
                # 使用遮罩进行混合：mask=1的地方使用cropped_image，mask=0的地方保持background
                blended_region = bg_region * (1 - mask_expanded) + cropped_image[:, :copy_height, :copy_width, :] * mask_expanded
                
                # 将混合后的区域放回输出图像
                output_image[:, y_min:y_min+copy_height, x_min:x_min+copy_width, :] = blended_region
        except Exception as e:
            # 如果遮罩合成失败，保持原始行为
            pass
        
        return (output_image, output_mask)

class ImageScaleRestoreyc:
    def __init__(self):
        self.NODE_NAME = 'ImageScaleRestore V2'

    @classmethod
    def INPUT_TYPES(cls):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        scale_by_list = ['by_scale', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "method": (method_mode,),
                "scale_by": (scale_by_list,),  # 是否按长边缩放
                "scale_by_length": ("INT", {"default": 1024, "min": 4, "max": 99999999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
                "original_size": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height",)
    FUNCTION = 'image_scale_restore'
    CATEGORY = 'YCNode/Image'

    def image_scale_restore(self, image, scale, method,
                            scale_by, scale_by_length,
                            mask = None,  original_size = None
                            ):

        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(l_images), len(l_masks))

        orig_width, orig_height = tensor2pil(l_images[0]).size
        if original_size is not None:
            target_width = original_size[0]
            target_height = original_size[1]
        else:
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            if scale_by == 'longest':
                if orig_width > orig_height:
                    target_width = scale_by_length
                    target_height = int(target_width * orig_height / orig_width)
                else:
                    target_height = scale_by_length
                    target_width = int(target_height * orig_width / orig_height)
            if scale_by == 'shortest':
                if orig_width < orig_height:
                    target_width = scale_by_length
                    target_height = int(target_width * orig_height / orig_width)
                else:
                    target_height = scale_by_length
                    target_width = int(target_height * orig_width / orig_height)
            if scale_by == 'width':
                target_width = scale_by_length
                target_height = int(target_width * orig_height / orig_width)
            if scale_by == 'height':
                target_height = scale_by_length
                target_width = int(target_height * orig_width / orig_height)
            if scale_by == 'total_pixel(kilo pixel)':
                r = orig_width / orig_height
                target_width = math.sqrt(r * scale_by_length * 1000)
                target_height = target_width / r
                target_width = int(target_width)
                target_height = int(target_height)
        if target_width < 4:
            target_width = 4
        if target_height < 4:
            target_height = 4
        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        for i in range(max_batch):

            _image = l_images[i] if i < len(l_images) else l_images[-1]

            _canvas = tensor2pil(_image).convert('RGB')
            ret_image = _canvas.resize((target_width, target_height), resize_sampler)
            ret_mask = Image.new('L', size=ret_image.size, color='white')
            if mask is not None:
                _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
                ret_mask = _mask.resize((target_width, target_height), resize_sampler)

            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(ret_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), [orig_width, orig_height], target_width, target_height,)

class YCMaskRatioPadCrop:
    """
    基于遮罩自动裁切到指定比例，必要时自动扩边并输出扩展区域遮罩
    """

    RATIO_MAP = {
        "1:1": (1, 1),
        "3:4": (3, 4),
        "4:3": (4, 3),
        "9:16": (9, 16),
        "16:9": (16, 9),
        "3:2": (3, 2),
        "2:3": (2, 3),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "ratio": (list(cls.RATIO_MAP.keys()), {"default": "1:1"}),
                "pad_mode": (PAD_MODES, {"default": "edge"}),
                "fill_color": ("STRING", {"default": "#ffffff", "multiline": False}),
                "extra_expand": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 1}),
                "padding_mode": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "ratio_mask", "pad_mask")
    FUNCTION = "execute"
    CATEGORY = "YCNode/Mask"

    def execute(self, image, mask, ratio, pad_mode, fill_color, extra_expand, padding_mode):
        if image.shape[0] != mask.shape[0]:
            raise ValueError("image 与 mask 的 batch 大小需要一致")

        ret_images = []
        ret_masks = []
        ret_pad_masks = []
        ret_original_masks = []

        for idx in range(image.shape[0]):
            img_tensor = image[idx]
            mask_tensor = mask[idx]

            pil_img = tensor2pil(torch.unsqueeze(img_tensor, 0))
            mask_np = self._prepare_mask(mask_tensor)

            processed_img, processed_mask, pad_mask_np, original_mask_np = self._process_single(
                pil_img, mask_np, ratio, pad_mode, fill_color, extra_expand, padding_mode
            )

            ret_images.append(pil2tensor(processed_img))
            ret_masks.append(torch.from_numpy(processed_mask).unsqueeze(0))
            ret_pad_masks.append(torch.from_numpy(pad_mask_np).unsqueeze(0))
            ret_original_masks.append(torch.from_numpy(original_mask_np).unsqueeze(0))

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_original_masks, dim=0),
            torch.cat(ret_masks, dim=0),
            torch.cat(ret_pad_masks, dim=0),
        )

    def _prepare_mask(self, mask_tensor):
        mask_np = mask_tensor.detach().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        return mask_np.astype(np.float32)

    def _process_single(self, image_pil, mask_np, ratio, pad_mode, fill_color, extra_expand, padding_mode):
        width, height = image_pil.size
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        coords = np.argwhere(binary_mask > 0)
        if coords.size == 0:
            min_y, min_x = 0, 0
            max_y, max_x = height - 1, width - 1
        else:
            min_y = int(coords[:, 0].min())
            max_y = int(coords[:, 0].max())
            min_x = int(coords[:, 1].min())
            max_x = int(coords[:, 1].max())

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        bbox_w = max(1, bbox_w)
        bbox_h = max(1, bbox_h)

        target_w_ratio, target_h_ratio = self.RATIO_MAP[ratio]
        target_ratio = target_w_ratio / target_h_ratio

        if (bbox_w / bbox_h) >= target_ratio:
            target_w = bbox_w
            target_h = int(math.ceil(target_w / target_ratio))
        else:
            target_h = bbox_h
            target_w = int(math.ceil(target_h * target_ratio))

        target_w = max(1, target_w)
        target_h = max(1, target_h)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        if padding_mode == "top":
            start_y = min_y
            end_y = start_y + target_h
            start_x = int(round(center_x - target_w / 2.0))
            end_x = start_x + target_w
        elif padding_mode == "bottom":
            end_y = max_y + 1
            start_y = end_y - target_h
            start_x = int(round(center_x - target_w / 2.0))
            end_x = start_x + target_w
        elif padding_mode == "left":
            start_x = min_x
            end_x = start_x + target_w
            start_y = int(round(center_y - target_h / 2.0))
            end_y = start_y + target_h
        elif padding_mode == "right":
            end_x = max_x + 1
            start_x = end_x - target_w
            start_y = int(round(center_y - target_h / 2.0))
            end_y = start_y + target_h
        else:
            start_x = int(round(center_x - target_w / 2.0))
            start_y = int(round(center_y - target_h / 2.0))
            end_x = start_x + target_w
            end_y = start_y + target_h

        pad_left = max(0, -start_x)
        pad_top = max(0, -start_y)
        pad_right = max(0, end_x - width)
        pad_bottom = max(0, end_y - height)

        padded_image = image_pil
        # 对原始mask进行相同的pad操作
        padded_mask = mask_np.copy()
        if pad_left or pad_top or pad_right or pad_bottom:
            padded_image = smart_pad_image(
                image_pil, pad_left, pad_top, pad_right, pad_bottom, pad_mode, fill_color
            )
            # 将mask转换为PIL图像进行pad操作
            mask_pil = Image.fromarray((padded_mask * 255).astype(np.uint8), mode='L')
            padded_mask_pil = smart_pad_image(
                mask_pil, pad_left, pad_top, pad_right, pad_bottom, pad_mode='constant', fill_color='#000000'
            )
            padded_mask = np.array(padded_mask_pil).astype(np.float32) / 255.0

        final_width, final_height = padded_image.size

        pad_indicator = np.zeros((final_height, final_width), dtype=np.float32)
        if pad_top > 0:
            pad_indicator[:pad_top, :] = 1.0
        if pad_bottom > 0:
            pad_indicator[-pad_bottom:, :] = 1.0
        if pad_left > 0:
            pad_indicator[:, :pad_left] = 1.0
        if pad_right > 0:
            pad_indicator[:, -pad_right:] = 1.0

        start_x += pad_left
        end_x += pad_left
        start_y += pad_top
        end_y += pad_top

        padded_width, padded_height = padded_image.size
        start_x = max(0, min(start_x, padded_width - target_w))
        start_y = max(0, min(start_y, padded_height - target_h))
        end_x = start_x + target_w
        end_y = start_y + target_h

        contact_left = start_x == 0
        contact_right = end_x == padded_width
        contact_top = start_y == 0
        contact_bottom = end_y == padded_height

        ratio_mask = np.zeros((final_height, final_width), dtype=np.float32)
        ratio_mask[start_y:end_y, start_x:end_x] = 1.0

        needs_extra_pad = contact_left or contact_right or contact_top or contact_bottom
        if extra_expand > 0 and needs_extra_pad:
            extra_sides = self._compute_extra_padding(
                extra_expand, contact_left, contact_top, contact_right, contact_bottom
            )
            if any(extra_sides.values()):
                padded_image = smart_pad_image(
                    padded_image,
                    extra_sides["left"],
                    extra_sides["top"],
                    extra_sides["right"],
                    extra_sides["bottom"],
                    pad_mode=pad_mode,
                    fill_color=fill_color,
                )
                ratio_mask = np.pad(
                    ratio_mask,
                    ((extra_sides["top"], extra_sides["bottom"]), (extra_sides["left"], extra_sides["right"])),
                    mode="constant",
                    constant_values=0.0,
                )
                pad_indicator = np.pad(
                    pad_indicator,
                    ((extra_sides["top"], extra_sides["bottom"]), (extra_sides["left"], extra_sides["right"])),
                    mode="constant",
                    constant_values=1.0,
                )
                # 对原始mask也进行相同的extra pad操作
                mask_pil = Image.fromarray((padded_mask * 255).astype(np.uint8), mode='L')
                padded_mask_pil = smart_pad_image(
                    mask_pil,
                    extra_sides["left"],
                    extra_sides["top"],
                    extra_sides["right"],
                    extra_sides["bottom"],
                    pad_mode='constant',
                    fill_color='#000000'
                )
                padded_mask = np.array(padded_mask_pil).astype(np.float32) / 255.0

        ratio_mask = np.clip(ratio_mask, 0.0, 1.0)
        pad_indicator = np.clip(pad_indicator, 0.0, 1.0)
        padded_mask = np.clip(padded_mask, 0.0, 1.0)

        return padded_image, ratio_mask, pad_indicator, padded_mask

    def _compute_extra_padding(self, extra_expand, contact_left, contact_top, contact_right, contact_bottom):
        extra_expand = max(0, int(extra_expand))
        pads = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        if extra_expand == 0:
            return pads

        if contact_top:
            pads["top"] = extra_expand
        if contact_bottom:
            pads["bottom"] = extra_expand
        if contact_left:
            pads["left"] = extra_expand
        if contact_right:
            pads["right"] = extra_expand

        return pads

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MaskCrop_YC": MaskCrop_YC,
    "MaskCropRestore_YC": MaskCropRestore_YC,
    "ImageScaleRestoreV2_YC": ImageScaleRestoreyc,
    "YCMaskRatioPadCrop": YCMaskRatioPadCrop,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCrop_YC": "MaskCrop_YC",
    "MaskCropRestore_YC": "MaskCropRestore_YC",
    "ImageScaleRestoreV2_YC": "ImageScaleRestore_YC",
    "YCMaskRatioPadCrop": "Mask Ratio Pad (YC)",
} 
