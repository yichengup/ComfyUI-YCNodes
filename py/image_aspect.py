import torch
import numpy as np
from PIL import Image
import cv2

def tensor2pil(t_image: torch.Tensor) -> Image:
    """将tensor转换为PIL图像"""
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image) -> torch.Tensor:
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parse_aspect_ratio(aspect_str: str) -> tuple:
    """
    解析宽高比字符串或具体尺寸
    
    Args:
        aspect_str: 宽高比字符串（如 "16:9"）或具体尺寸（如 "768x1024"）或带标注（如 "1792x2400 3:4"）
        
    Returns:
        (width_value, height_value, is_absolute_size, label_ratio) 元组
        - is_absolute_size: True表示具体尺寸，False表示比例
        - label_ratio: 用户标注的比例字符串（如果有），否则为None
    """
    aspect_str = aspect_str.strip()
    label_ratio = None
    
    # 检查是否有空格分隔的标注格式（如 "1792x2400 3:4"）
    if ' ' in aspect_str:
        parts = aspect_str.split(' ', 1)
        aspect_str = parts[0].strip()
        label_ratio = parts[1].strip()
    
    # 尝试解析 "宽x高" 或 "宽:高" 格式
    for separator in ['x', 'X', '*', '×', ':']:
        if separator in aspect_str:
            parts = aspect_str.split(separator)
            if len(parts) == 2:
                try:
                    w = float(parts[0].strip())
                    h = float(parts[1].strip())
                    if w > 0 and h > 0:
                        # 判断是具体尺寸还是比例
                        # 如果宽度和高度都 >= 64，判定为具体尺寸
                        is_absolute = (w >= 64 and h >= 64)
                        return (w, h, is_absolute, label_ratio)
                except ValueError:
                    pass
    
    # 尝试解析小数格式（只能是比例）
    try:
        ratio = float(aspect_str)
        if ratio > 0:
            return (ratio, 1.0, False, label_ratio)
    except ValueError:
        pass
    
    # 默认返回 16:9 比例
    return (16.0, 9.0, False, label_ratio)


class ImageAspectExpand:
    """
    图像比例扩展节点
    根据自定义比例扩展图像，保证原图完整显示
    支持多行输入，自动选择最接近的比例
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratios": ("STRING", {
                    "default": "16:9\n9:16\n4:3\n3:4\n1:1", 
                    "multiline": True,
                    "tooltip": "比例或尺寸列表，每行一个。格式: 比例(如4:3) 或 具体尺寸(如768x1024)。多行时自动选择最接近的"
                }),
                "fill_color": ("STRING", {"default": "#000000", "tooltip": "填充颜色（十六进制）"}),
                "alignment": (["center", "top", "bottom", "left", "right", 
                              "top_left", "top_right", "bottom_left", "bottom_right"],
                             {"default": "center", "tooltip": "原图对齐方式"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("expanded_image", "original_mask", "width", "height", "selected_ratio")
    FUNCTION = "expand_aspect"
    CATEGORY = "YCNode/Image"

    def expand_aspect(self, image: torch.Tensor, aspect_ratios: str,
                     fill_color: str, alignment: str) -> tuple:
        """
        根据比例扩展图像，保证原图完整显示
        
        Args:
            image: 输入图像
            aspect_ratios: 比例列表（多行）或单个比例（单行）
            fill_color: 填充颜色
            alignment: 对齐方式
            
        Returns:
            (expanded_image, original_mask, width, height, selected_ratio)
        """
        try:
            # 解析所有比例
            ratio_lines = [line.strip() for line in aspect_ratios.strip().split('\n') if line.strip()]
            
            if not ratio_lines:
                raise ValueError("至少需要提供一个比例")
            
            # 解析填充颜色
            fill_rgb = self._hex_to_rgb(fill_color)
            
            batch_expanded = []
            batch_masks = []
            output_width = 0
            output_height = 0
            selected_ratio_str = ""
            
            # 处理批次中的每张图像
            for img_tensor in image:
                # 转换为PIL图像
                img_pil = tensor2pil(img_tensor)
                orig_width, orig_height = img_pil.size
                orig_aspect = orig_width / orig_height
                
                # 选择最接近的比例或尺寸
                label_ratio = None
                if len(ratio_lines) == 1:
                    # 单行：直接使用
                    target_w, target_h, is_absolute, label_ratio = parse_aspect_ratio(ratio_lines[0])
                else:
                    # 多行：选择最接近原图比例的
                    best_ratio_str = ratio_lines[0]
                    best_w, best_h, best_is_absolute, best_label = parse_aspect_ratio(best_ratio_str)
                    best_aspect = best_w / best_h
                    min_diff = abs(best_aspect - orig_aspect)
                    
                    for ratio_str in ratio_lines[1:]:
                        w, h, is_abs, lbl = parse_aspect_ratio(ratio_str)
                        aspect = w / h
                        diff = abs(aspect - orig_aspect)
                        
                        if diff < min_diff:
                            min_diff = diff
                            best_w = w
                            best_h = h
                            best_is_absolute = is_abs
                            best_aspect = aspect
                            best_label = lbl
                    
                    target_w = best_w
                    target_h = best_h
                    is_absolute = best_is_absolute
                    label_ratio = best_label
                
                target_aspect = target_w / target_h
                
                # 确定输出的比例字符串
                if label_ratio:
                    # 如果有用户标注的比例，直接使用
                    selected_ratio_str = label_ratio
                elif is_absolute:
                    # 具体尺寸且无标注，使用常见比例匹配
                    selected_ratio_str = self._match_common_ratio(int(target_w), int(target_h))
                else:
                    # 纯比例输入，格式化输出
                    selected_ratio_str = self._format_aspect_ratio(int(target_w), int(target_h))
                
                # 根据是否为具体尺寸，采用不同的处理方式
                if is_absolute:
                    # 具体尺寸模式：先缩放到目标尺寸，再扩展
                    target_width = int(target_w)
                    target_height = int(target_h)
                    
                    # 计算缩放比例（保持宽高比，确保能放入目标尺寸）
                    scale_w = target_width / orig_width
                    scale_h = target_height / orig_height
                    scale = min(scale_w, scale_h)  # 选择较小的缩放比例，确保不超出
                    
                    # 计算缩放后的尺寸
                    scaled_width = int(orig_width * scale)
                    scaled_height = int(orig_height * scale)
                    
                    # 缩放图像
                    img_pil = img_pil.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                    
                    # 更新原图尺寸为缩放后的尺寸
                    orig_width = scaled_width
                    orig_height = scaled_height
                    
                    # 目标尺寸就是指定的具体尺寸
                    new_width = target_width
                    new_height = target_height
                    
                    print(f"[ImageAspectExpand] 具体尺寸模式: 缩放到 {scaled_width}x{scaled_height}, 扩展到 {new_width}x{new_height}")
                else:
                    # 比例模式：只扩展，不缩放（原有逻辑）
                    new_width = orig_width
                    new_height = int(new_width / target_aspect)
                    
                    # 如果计算出的高度小于原高度，说明需要扩展宽度
                    if new_height < orig_height:
                        new_height = orig_height
                        new_width = int(new_height * target_aspect)
                    
                    print(f"[ImageAspectExpand] 比例模式: 扩展到 {new_width}x{new_height}")
                
                # 创建新画布
                expanded_img = Image.new('RGB', (new_width, new_height), fill_rgb)
                
                # 计算原图粘贴位置
                paste_x, paste_y = self._calculate_paste_position(
                    orig_width, orig_height, new_width, new_height, alignment
                )
                
                # 粘贴原图
                expanded_img.paste(img_pil, (paste_x, paste_y))
                
                # 创建遮罩（原图区域为白色，扩展区域为黑色）
                mask = Image.new('L', (new_width, new_height), 0)
                mask_draw = Image.new('L', (orig_width, orig_height), 255)
                mask.paste(mask_draw, (paste_x, paste_y))
                
                # 转换为tensor
                expanded_tensor = pil2tensor(expanded_img)
                mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)
                
                batch_expanded.append(expanded_tensor)
                batch_masks.append(mask_tensor)
                
                output_width = new_width
                output_height = new_height
                
                mode_str = "具体尺寸" if is_absolute else "比例"
                print(f"[ImageAspectExpand] 模式: {mode_str}")
                print(f"[ImageAspectExpand] 可选项: {len(ratio_lines)} 个")
                print(f"[ImageAspectExpand] 选择: {selected_ratio_str} ({target_aspect:.3f})")
                print(f"[ImageAspectExpand] 最终尺寸: {new_width}x{new_height}")
                print(f"[ImageAspectExpand] 对齐方式: {alignment}")
            
            # 合并批次
            final_expanded = torch.cat(batch_expanded, dim=0)
            final_masks = torch.stack(batch_masks, dim=0)
            
            return (final_expanded, final_masks, output_width, output_height, selected_ratio_str)
            
        except Exception as e:
            raise ValueError(f"图像比例扩展失败: {str(e)}")

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (0, 0, 0)
    
    def _match_common_ratio(self, width: int, height: int) -> str:
        """
        匹配常见比例
        如果尺寸接近常见比例（误差 < 1%），返回常见比例字符串
        否则返回简化后的比例
        """
        actual_ratio = width / height
        
        # 常见比例列表 (比例值, 显示字符串)
        common_ratios = [
            (1.0, "1:1"),
            (4/3, "4:3"),
            (3/4, "3:4"),
            (16/9, "16:9"),
            (9/16, "9:16"),
            (21/9, "21:9"),
            (9/21, "9:21"),
            (3/2, "3:2"),
            (2/3, "2:3"),
            (5/4, "5:4"),
            (4/5, "4:5"),
        ]
        
        # 检查是否接近常见比例（误差 < 1%）
        for ratio_value, ratio_str in common_ratios:
            if abs(actual_ratio - ratio_value) / ratio_value < 0.01:
                return ratio_str
        
        # 如果不接近常见比例，返回简化后的比例
        return self._format_aspect_ratio(width, height)
    
    def _format_aspect_ratio(self, width: int, height: int) -> str:
        """
        格式化宽高比为字符串
        尝试简化为常见比例格式
        """
        # 计算最大公约数
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        divisor = gcd(width, height)
        simplified_w = width // divisor
        simplified_h = height // divisor
        
        # 如果简化后的数字合理（小于100），使用简化格式
        if simplified_w <= 100 and simplified_h <= 100:
            return f"{simplified_w}:{simplified_h}"
        else:
            # 否则使用原始尺寸作为比例
            return f"{width}:{height}"
    
    def _calculate_paste_position(self, orig_w: int, orig_h: int, 
                                  new_w: int, new_h: int, alignment: str) -> tuple:
        """计算原图在新画布上的粘贴位置"""
        if alignment == "center":
            x = (new_w - orig_w) // 2
            y = (new_h - orig_h) // 2
        elif alignment == "top":
            x = (new_w - orig_w) // 2
            y = 0
        elif alignment == "bottom":
            x = (new_w - orig_w) // 2
            y = new_h - orig_h
        elif alignment == "left":
            x = 0
            y = (new_h - orig_h) // 2
        elif alignment == "right":
            x = new_w - orig_w
            y = (new_h - orig_h) // 2
        elif alignment == "top_left":
            x = 0
            y = 0
        elif alignment == "top_right":
            x = new_w - orig_w
            y = 0
        elif alignment == "bottom_left":
            x = 0
            y = new_h - orig_h
        elif alignment == "bottom_right":
            x = new_w - orig_w
            y = new_h - orig_h
        else:
            x = (new_w - orig_w) // 2
            y = (new_h - orig_h) // 2
        
        return (x, y)

class ImageSizeMatcher:
    """
    图像尺寸匹配器
    根据输入图像的尺寸比例，从预设的尺寸列表中选择最接近的尺寸输出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "preset_sizes": ("STRING", {
                    "multiline": True, 
                    "default": "512x768\n512x512\n1024x1024\n768x512\n1024x768"
                }),  # 预设尺寸列表，每行一个尺寸
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "size_string", "aspect_ratio")
    FUNCTION = "match_size"
    CATEGORY = "YCNode/Image"
    
    def match_size(self, image: torch.Tensor, preset_sizes: str) -> tuple:
        """
        匹配最接近的预设尺寸
        
        Args:
            image: 输入图像张量 [batch, height, width, channels]
            preset_sizes: 预设尺寸字符串，格式如 "512x768\n1024x1024"
            
        Returns:
            (width, height, size_string, aspect_ratio_string)
        """
        try:
            # 获取输入图像的尺寸
            # ComfyUI图像格式: [batch, height, width, channels]
            batch, img_height, img_width, channels = image.shape
            img_aspect_ratio = img_width / img_height
            
            # 解析预设尺寸列表
            preset_list = []
            for line in preset_sizes.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # 支持多种分隔符: x, X, *, ×
                for separator in ['x', 'X', '*', '×', ',']:
                    if separator in line:
                        parts = line.split(separator)
                        if len(parts) == 2:
                            try:
                                width = int(parts[0].strip())
                                height = int(parts[1].strip())
                                if width > 0 and height > 0:
                                    preset_list.append((width, height))
                                    break
                            except ValueError:
                                continue
            
            # 检查是否有有效的预设尺寸
            if not preset_list:
                raise ValueError("没有有效的预设尺寸。请使用格式: 宽x高 (例如: 512x768)")
            
            # 找到比例最接近的预设尺寸
            best_match = None
            min_ratio_diff = float('inf')
            
            for width, height in preset_list:
                preset_aspect_ratio = width / height
                ratio_diff = abs(preset_aspect_ratio - img_aspect_ratio)
                
                if ratio_diff < min_ratio_diff:
                    min_ratio_diff = ratio_diff
                    best_match = (width, height)
            
            # 返回最佳匹配的尺寸
            width, height = best_match
            size_string = f"{width}x{height}"
            
            # 计算比例并格式化为字符串（与ImageAspectExpand一致）
            aspect_ratio_value = width / height
            # 尝试简化比例
            aspect_ratio_string = self._format_aspect_ratio(width, height)
            
            print(f"[ImageSizeMatcher] 输入图像尺寸: {img_width}x{img_height} (比例: {img_aspect_ratio:.3f})")
            print(f"[ImageSizeMatcher] 匹配的预设尺寸: {size_string} (比例: {aspect_ratio_string})")
            print(f"[ImageSizeMatcher] 比例差异: {min_ratio_diff:.3f}")
            
            return (width, height, size_string, aspect_ratio_string)
            
        except Exception as e:
            raise ValueError(f"尺寸匹配失败: {str(e)}")
    
    def _format_aspect_ratio(self, width: int, height: int) -> str:
        """
        格式化宽高比为字符串
        尝试简化为常见比例格式
        """
        # 计算最大公约数
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        divisor = gcd(width, height)
        simplified_w = width // divisor
        simplified_h = height // divisor
        
        # 如果简化后的数字合理（小于100），使用简化格式
        if simplified_w <= 100 and simplified_h <= 100:
            return f"{simplified_w}:{simplified_h}"
        else:
            # 否则使用原始尺寸
            return f"{width}:{height}"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """验证输入参数"""
        if "preset_sizes" in kwargs:
            preset_sizes = kwargs["preset_sizes"].strip()
            if not preset_sizes:
                return "预设尺寸列表不能为空"
            
            # 验证至少有一个有效的尺寸
            valid_count = 0
            for line in preset_sizes.split('\n'):
                line = line.strip()
                if not line:
                    continue
                for separator in ['x', 'X', '*', '×', ',']:
                    if separator in line:
                        parts = line.split(separator)
                        if len(parts) == 2:
                            try:
                                width = int(parts[0].strip())
                                height = int(parts[1].strip())
                                if width > 0 and height > 0:
                                    valid_count += 1
                                    break
                            except ValueError:
                                continue
            
            if valid_count == 0:
                return "没有有效的预设尺寸。请使用格式: 宽x高 (例如: 512x768)"
        
        return True

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageAspectExpand": ImageAspectExpand,
    "ImageSizeMatcher": ImageSizeMatcher
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAspectExpand": "Image Aspect Expand",
    "ImageSizeMatcher": "Image Size Matcher"
}

