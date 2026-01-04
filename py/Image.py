import torch
import numpy as np
from PIL import Image
import cv2
import math

# yicheng author
MAX_RESOLUTION = 16384
# 填充模式列表
PAD_MODES = [
    'edge',      # 边缘颜色填充
    'reflect',   # 镜像反射填充
    'symmetric', # 对称填充
    'wrap',      # 循环填充
    'constant'   # 固定颜色填充
]

def pil2tensor(image):
    """PIL图像转tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    """tensor转PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def hex_to_rgb(hex_color):
    """十六进制颜色转RGB元组"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """RGB元组转十六进制颜色"""
    return '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])

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

def create_pad_mask(original_width, original_height, left, top, right, bottom, invert_mask=False):
    """创建扩图遮罩，可选择遮罩方向
    
    Args:
        original_width: 原始图像宽度
        original_height: 原始图像高度
        left: 左边填充像素
        top: 上边填充像素
        right: 右边填充像素
        bottom: 下边填充像素
        invert_mask: 是否反向遮罩
            - False: 原始图像区域为白色(1.0)，填充区域为黑色(0.0)
            - True: 原始图像区域为黑色(0.0)，填充区域为白色(1.0)
    """
    # 计算扩图后的尺寸
    new_width = original_width + left + right
    new_height = original_height + top + bottom
    
    # 根据反向设置决定默认颜色
    default_color = 255 if invert_mask else 0
    fill_color = 0 if invert_mask else 255
    
    # 创建PIL遮罩图像
    mask_image = Image.new('L', (new_width, new_height), default_color)
    
    # 在原始图像位置绘制填充色矩形
    # 原始图像在扩图后的位置是 (left, top) 到 (left + original_width, top + original_height)
    mask_image.paste(fill_color, (left, top, left + original_width, top + original_height))
    
    # 转换为numpy数组并归一化到0.0-1.0
    mask_np = np.array(mask_image).astype(np.float32) / 255.0
    
    # 转换为torch tensor并添加批次维度
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, height, width]
    
    return mask_tensor


def smart_crop_image(image, left, top, right, bottom):
    """智能图像裁剪函数
    
    Args:
        image: PIL图像
        left: 左边裁剪像素数
        top: 上边裁剪像素数
        right: 右边裁剪像素数
        bottom: 下边裁剪像素数
    """
    # 获取原始图像尺寸
    original_width, original_height = image.size
    
    # 计算裁剪后的尺寸
    new_width = original_width - left - right
    new_height = original_height - top - bottom
    
    # 确保裁剪后的尺寸大于0
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"裁剪后尺寸无效: {new_width}x{new_height}")
    
    # 计算裁剪区域
    crop_left = left
    crop_top = top
    crop_right = original_width - right
    crop_bottom = original_height - bottom
    
    # 执行裁剪
    result = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    return result

def create_crop_mask(original_width, original_height, left, top, right, bottom, invert_mask=False):
    """创建裁剪遮罩，可选择遮罩方向
    
    Args:
        original_width: 原始图像宽度
        original_height: 原始图像高度
        left: 左边裁剪像素
        top: 上边裁剪像素
        right: 右边裁剪像素
        bottom: 下边裁剪像素
        invert_mask: 是否反向遮罩
            - False: 保留区域为白色(1.0)，裁剪区域为黑色(0.0)
            - True: 保留区域为黑色(0.0)，裁剪区域为白色(1.0)
    """
    # 计算裁剪后的尺寸
    new_width = original_width - left - right
    new_height = original_height - top - bottom
    
    # 确保尺寸有效
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"裁剪后尺寸无效: {new_width}x{new_height}")
    
    # 根据反向设置决定默认颜色
    default_color = 255 if invert_mask else 0
    fill_color = 0 if invert_mask else 255
    
    # 创建PIL遮罩图像（使用原始尺寸）
    mask_image = Image.new('L', (original_width, original_height), default_color)
    
    # 在保留区域绘制填充色矩形
    # 保留区域是 (left, top) 到 (left + new_width, top + new_height)
    mask_image.paste(fill_color, (left, top, left + new_width, top + new_height))
    
    # 转换为numpy数组并归一化到0.0-1.0
    mask_np = np.array(mask_image).astype(np.float32) / 255.0
    
    # 转换为torch tensor并添加批次维度
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, height, width]
    
    return mask_tensor

# yicheng author
class YCImageSmartPad:
    """智能图像扩图节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "top": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "right": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "bottom": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "pad_mode": (PAD_MODES, {"default": "edge"}),
                "fill_color": ("STRING", {"default": "#fffddd", "multiline": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "smart_pad"
    CATEGORY = "YCNode/Image"

    def smart_pad(self, image, left, top, right, bottom, pad_mode, fill_color, invert_mask):
        """智能扩图处理"""
        ret_images = []
        ret_masks = []
        
        # 处理batch
        for img in image:
            # 转换为PIL图像
            pil_img = tensor2pil(torch.unsqueeze(img, 0))
            
            # 获取原始图像尺寸
            original_width, original_height = pil_img.size
            
            # 执行智能填充
            result_pil = smart_pad_image(
                pil_img, left, top, right, bottom, pad_mode, fill_color
            )
            
            # 转换回tensor
            result_tensor = pil2tensor(result_pil)
            ret_images.append(result_tensor)
            
            # 创建对应的遮罩
            mask_tensor = create_pad_mask(original_width, original_height, left, top, right, bottom, invert_mask)
            ret_masks.append(mask_tensor)
        
        # 返回结果
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))

# yicheng author
class YCImageSmartCrop:
    """智能图像裁剪节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "top": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "right": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "bottom": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "smart_crop"
    CATEGORY = "YCNode/Image"

    def smart_crop(self, image, left, top, right, bottom, invert_mask):
        """智能裁剪处理"""
        ret_images = []
        ret_masks = []
        
        # 处理batch
        for img in image:
            # 转换为PIL图像
            pil_img = tensor2pil(torch.unsqueeze(img, 0))
            
            # 获取原始图像尺寸
            original_width, original_height = pil_img.size
            
            # 执行智能裁剪
            result_pil = smart_crop_image(pil_img, left, top, right, bottom)
            
            # 转换回tensor
            result_tensor = pil2tensor(result_pil)
            ret_images.append(result_tensor)
            
            # 创建对应的遮罩（使用原始尺寸）
            mask_tensor = create_crop_mask(original_width, original_height, left, top, right, bottom, invert_mask)
            ret_masks.append(mask_tensor)
        
        # 返回结果
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))

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

class YCImageTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01, }),
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "tile_width", "tile_height", "overlap_x", "overlap_y",)
    FUNCTION = "execute"
    CATEGORY = "YCNode/Image"

    def execute(self, image, rows, cols, overlap, overlap_x, overlap_y):
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        h = tile_h * rows
        w = tile_w * cols
        overlap_h = int(tile_h * overlap) + overlap_y
        overlap_w = int(tile_w * overlap) + overlap_x

        # max overlap is half of the tile size
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0
        
        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h + overlap_h
                x2 = x1 + tile_w + overlap_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h - overlap_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w - overlap_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        return(tiles, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h,)

class YCImageUntile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "YCNode/Image"

    def execute(self, tiles, overlap_x, overlap_y, rows, cols):
        tile_h, tile_w = tiles.shape[1:3]
        tile_h -= overlap_y
        tile_w -= overlap_x
        out_w = cols * tile_w
        out_h = rows * tile_h

        out = torch.zeros((1, out_h, out_w, tiles.shape[3]), device=tiles.device, dtype=tiles.dtype)

        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_y
                if j > 0:
                    x1 -= overlap_x

                y2 = y1 + tile_h + overlap_y
                x2 = x1 + tile_w + overlap_x

                if y2 > out_h:
                    y2 = out_h
                    y1 = y2 - tile_h - overlap_y
                if x2 > out_w:
                    x2 = out_w
                    x1 = x2 - tile_w - overlap_x
                
                mask = torch.ones((1, tile_h+overlap_y, tile_w+overlap_x), device=tiles.device, dtype=tiles.dtype)

                # feather the overlap on top
                if i > 0 and overlap_y > 0:
                    mask[:, :overlap_y, :] *= torch.linspace(0, 1, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on bottom
                #if i < rows - 1:
                #    mask[:, -overlap_y:, :] *= torch.linspace(1, 0, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on left
                if j > 0 and overlap_x > 0:
                    mask[:, :, :overlap_x] *= torch.linspace(0, 1, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                # feather the overlap on right
                #if j < cols - 1:
                #    mask[:, :, -overlap_x:] *= torch.linspace(1, 0, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
                tile = tiles[i * cols + j] * mask
                out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile
        return(out, )

class YC_MaskColorOverlay:
    """
    将遮罩的白色区域应用指定颜色，并设置透明度
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#FF0000", "multiline": False}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_overlay"
    CATEGORY = "YCNode/Image"
    
    def apply_color_overlay(self, image, mask, color, opacity):
        # 将十六进制颜色转换为RGB
        hex_color = color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # 转换为numpy数组
        img_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # 确保mask是二维的
        if len(mask_np.shape) == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np.squeeze(-1)
        
        # 创建输出图像
        result = img_np.copy()
        
        # 对每个批次处理
        for b in range(img_np.shape[0]):
            # 创建颜色层
            color_layer = np.zeros_like(img_np[b])
            color_layer[:, :, 0] = rgb_color[0] / 255.0  # R
            color_layer[:, :, 1] = rgb_color[1] / 255.0  # G
            color_layer[:, :, 2] = rgb_color[2] / 255.0  # B
            
            # 应用遮罩
            mask_3d = np.stack([mask_np[b]] * 3, axis=-1)
            
            # 混合原图和颜色层
            result[b] = img_np[b] * (1 - mask_3d * opacity) + color_layer * mask_3d * opacity
        
        # 转换回torch张量
        result_tensor = torch.from_numpy(result)
        
        return (result_tensor,)

class ycImageTilem:
    """图像拼铺节点 - 将一张图拼铺成多行多列的大图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "cols": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "spacing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiled_image",)
    FUNCTION = "tile_image"
    CATEGORY = "YCNode/Image"
    
    def tile_image(self, image, rows, cols, spacing, background_color):
        """
        将输入图像拼铺成指定行列数的大图
        
        Args:
            image: 输入图像 (torch tensor)
            rows: 行数
            cols: 列数
            spacing: 间距（像素）
            background_color: 背景颜色（十六进制格式，如 #FFFFFF）
        
        Returns:
            拼铺后的大图
        """
        # 转换背景颜色
        bg_color = hex_to_rgb(background_color)
        
        # 处理batch，取第一张图
        if image.dim() == 4:
            input_image = image[0]
        else:
            input_image = image
        
        # 转换为PIL图像
        pil_image = tensor2pil(input_image)
        img_width, img_height = pil_image.size
        
        # 计算最终画布尺寸
        # 总宽度 = 列数 * 图片宽度 + (列数 - 1) * 间距
        # 总高度 = 行数 * 图片高度 + (行数 - 1) * 间距
        canvas_width = cols * img_width + (cols - 1) * spacing
        canvas_height = rows * img_height + (rows - 1) * spacing
        
        # 创建画布
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
        
        # 拼铺图像
        for row in range(rows):
            for col in range(cols):
                # 计算当前图像在画布中的位置
                x = col * (img_width + spacing)
                y = row * (img_height + spacing)
                
                # 将图像粘贴到画布上
                canvas.paste(pil_image, (x, y))
        
        # 转换回torch tensor
        result_tensor = pil2tensor(canvas)
        
        return (result_tensor,)


# 注册所有节点
NODE_CLASS_MAPPINGS = {
    "YCImageSmartPad": YCImageSmartPad,
    "YCImageSmartCrop": YCImageSmartCrop,
    "ImageMirror": ImageMirror,
    "ImageRotate": ImageRotate,
    "ImageMosaic": ImageMosaic,
    "YCImageTile": YCImageTile,
    "YCImageUntile": YCImageUntile,
    "YC_MaskColorOverlay": YC_MaskColorOverlay,
    "ycImageTilem": ycImageTilem
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YCImageSmartPad": "Yc Image Smart Pad",
    "YCImageSmartCrop": "Yc Image Smart Crop",
    "ImageMirror": "Image Mirror",
    "ImageRotate": "Image Rotate", 
    "ImageMosaic": "Image Mosaic",
    "YCImageTile": "YC Image Tile",
    "YCImageUntile": "YC Image Untile",
    "YC_MaskColorOverlay":"YC Mask Color Overlay",
    "ycImageTilem": "Image Tile merg"
} 
# yicheng author
