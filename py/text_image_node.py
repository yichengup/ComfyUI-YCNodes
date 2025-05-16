import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math

class TextImageGenerator:
    """
    生成文本图像和对应遮罩的节点
    支持中英文、水平垂直排版，可调整字体大小、颜色、位置等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取字体目录中所有ttf和otf文件
        font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font")
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)
            
        fonts = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
        if not fonts:
            # 如果没有字体文件，添加默认选项
            fonts = ["NotoSansSC-Regular.ttf"]
            
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "示例文本\nSample Text"}),
                "canvas_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "font_size": ("INT", {"default": 72, "min": 8, "max": 500, "step": 1}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "alignment": (["left", "center", "right"], {"default": "center"}),
                "vertical_alignment": (["top", "middle", "bottom"], {"default": "middle"}),
                "x_position": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "background_color": ("COLOR", {"default": "#000000"}),
                "text_color": ("COLOR", {"default": "#ffffff"}),
                "background_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "font": (fonts, ),
                "letter_spacing": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 2.0, "step": 0.01}),
                "word_spacing": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "line_spacing": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate_text_image"
    CATEGORY = "YCNode/Text"
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB值"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def generate_text_image(self, text, canvas_width, canvas_height, font_size, 
                         orientation, alignment, vertical_alignment,
                         x_position, y_position, background_color, text_color,
                         background_alpha, text_alpha, font,
                         letter_spacing, word_spacing, line_spacing):
        """生成文本图像和对应的遮罩"""
        
        # 转换颜色
        bg_color = self.hex_to_rgb(background_color)
        txt_color = self.hex_to_rgb(text_color)
        
        # 创建RGBA图像
        bg_color_with_alpha = bg_color + (int(background_alpha * 255),)
        image = Image.new('RGBA', (canvas_width, canvas_height), bg_color_with_alpha)
        draw = ImageDraw.Draw(image)
        
        # 创建文本遮罩图像 (黑色背景，白色文本)
        mask_image = Image.new('L', (canvas_width, canvas_height), 0)
        mask_draw = ImageDraw.Draw(mask_image)
        
        # 加载字体
        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "font", font)
        if not os.path.exists(font_path):
            # 如果字体文件不存在，尝试使用系统默认字体
            try:
                font_obj = ImageFont.truetype(font, font_size)
            except:
                font_obj = ImageFont.load_default()
                font_size = 12  # 默认字体可能较小
        else:
            font_obj = ImageFont.truetype(font_path, font_size)
        
        # 处理文本行
        lines = text.splitlines()
        
        # 计算文本尺寸和位置
        if orientation == "horizontal":
            # 水平排列文本
            line_heights = []
            line_widths = []
            
            # 计算字母间距的基础单位（像素）
            letter_spacing_px = font_size * letter_spacing
            # 计算单词间距的基础单位（像素）
            space_width = draw.textlength(" ", font=font_obj)
            word_spacing_px = space_width * (word_spacing - 1)  # 额外的空格宽度
            
            for line in lines:
                # 计算行宽度（包括字间距和词间距）
                if any(ord(c) < 128 for c in line):  # 包含ASCII字符
                    # 对于英文文本，计算每个字符的宽度和单词间距
                    total_width = 0
                    words = line.split()
                    
                    for i, word in enumerate(words):
                        # 单词内字符
                        word_width = 0
                        for j, char in enumerate(word):
                            char_width = draw.textlength(char, font=font_obj)
                            word_width += char_width
                            if j < len(word) - 1:  # 不是单词的最后一个字符
                                word_width += letter_spacing_px
                        
                        total_width += word_width
                        
                        # 单词间添加空格和额外的单词间距
                        if i < len(words) - 1:  # 不是最后一个单词
                            total_width += space_width + word_spacing_px
                    
                    line_width = total_width
                else:
                    # 对于纯中文文本，字符间添加字间距
                    total_width = 0
                    for i, char in enumerate(line):
                        char_width = draw.textlength(char, font=font_obj)
                        total_width += char_width
                        if i < len(line) - 1:  # 不是行的最后一个字符
                            total_width += letter_spacing_px
                    
                    line_width = total_width
                
                bbox = font_obj.getbbox(line)
                line_height = bbox[3] - bbox[1]
                
                line_heights.append(line_height)
                line_widths.append(line_width)
            
            # 计算行间距
            line_spacing_px = font_size * line_spacing
            total_height = sum(line_heights) + (len(lines) - 1) * line_spacing_px
            max_width = max(line_widths) if line_widths else 0
            
            # 计算起始位置
            if alignment == "left":
                x_start = x_position
            elif alignment == "center":
                x_start = x_position + (canvas_width - max_width) // 2
            else:  # right
                x_start = x_position + canvas_width - max_width
            
            if vertical_alignment == "top":
                y_start = y_position
            elif vertical_alignment == "middle":
                y_start = y_position + (canvas_height - total_height) // 2
            else:  # bottom
                y_start = y_position + canvas_height - total_height
            
            # 绘制文本
            y = y_start
            for i, line in enumerate(lines):
                x = x_start
                
                if any(ord(c) < 128 for c in line):  # 包含ASCII字符
                    # 对于英文文本，处理单词和字符
                    words = line.split()
                    for word_idx, word in enumerate(words):
                        # 绘制单词中的每个字符
                        for j, char in enumerate(word):
                            char_width = draw.textlength(char, font=font_obj)
                            
                            # 绘制到图像和遮罩
                            txt_color_with_alpha = txt_color + (int(text_alpha * 255),)
                            draw.text((x, y), char, fill=txt_color_with_alpha, font=font_obj)
                            mask_draw.text((x, y), char, fill=255, font=font_obj)
                            
                            x += char_width
                            if j < len(word) - 1:  # 不是单词的最后一个字符
                                x += letter_spacing_px
                        
                        # 添加单词间距
                        if word_idx < len(words) - 1:  # 不是最后一个单词
                            x += space_width + word_spacing_px
                else:
                    # 对于纯中文文本，逐字符绘制并添加字间距
                    for j, char in enumerate(line):
                        # 绘制到图像和遮罩
                        txt_color_with_alpha = txt_color + (int(text_alpha * 255),)
                        draw.text((x, y), char, fill=txt_color_with_alpha, font=font_obj)
                        mask_draw.text((x, y), char, fill=255, font=font_obj)
                        
                        # 计算下一个字符的位置
                        char_width = draw.textlength(char, font=font_obj)
                        x += char_width
                        
                        # 添加字间距
                        if j < len(line) - 1:  # 不是行的最后一个字符
                            x += letter_spacing_px
                
                # 计算下一行的位置（使用自定义行间距）
                y += line_heights[i] + line_spacing_px
                
        else:  # vertical
            # 垂直排列文本
            char_heights = []
            max_char_width = 0
            
            # 将所有文本合并为一个字符列表
            all_chars = []
            for line in lines:
                all_chars.extend(list(line))
                if line != lines[-1]:  # 如果不是最后一行，添加换行符
                    all_chars.append('\n')
            
            # 测量每个字符的尺寸
            for char in all_chars:
                if char == '\n':
                    # 换行符使用行间距
                    char_heights.append(font_size * line_spacing)
                    continue
                    
                bbox = font_obj.getbbox(char)
                char_height = bbox[3] - bbox[1]
                char_width = bbox[2] - bbox[0]
                
                char_heights.append(char_height)
                max_char_width = max(max_char_width, char_width)
            
            # 计算字符间距（垂直方向）
            letter_spacing_px = font_size * letter_spacing
            total_height = sum(char_heights) + (len(all_chars) - 1) * letter_spacing_px
            
            # 计算起始位置
            if alignment == "left":
                x_start = x_position
            elif alignment == "center":
                x_start = x_position + (canvas_width - max_char_width) // 2
            else:  # right
                x_start = x_position + canvas_width - max_char_width
            
            if vertical_alignment == "top":
                y_start = y_position
            elif vertical_alignment == "middle":
                y_start = y_position + (canvas_height - total_height) // 2
            else:  # bottom
                y_start = y_position + canvas_height - total_height
            
            # 绘制文本
            y = y_start
            prev_char = None
            for char in all_chars:
                if char == '\n':
                    # 换行符使用行间距
                    y += font_size * line_spacing
                    prev_char = None
                    continue
                
                # 计算特殊单词间距
                extra_spacing = 0
                if prev_char == ' ' and char != ' ':
                    # 单词之间的间距
                    extra_spacing = (word_spacing - 1) * font_size * 0.5
                    
                # 绘制到图像和遮罩
                txt_color_with_alpha = txt_color + (int(text_alpha * 255),)
                draw.text((x_start, y + extra_spacing), char, fill=txt_color_with_alpha, font=font_obj)
                mask_draw.text((x_start, y + extra_spacing), char, fill=255, font=font_obj)
                
                # 计算下一个字符的位置
                bbox = font_obj.getbbox(char)
                char_height = bbox[3] - bbox[1]
                y += char_height + letter_spacing_px
                
                prev_char = char
        
        # 将RGBA图像转换为RGB
        rgb_image = Image.new('RGB', image.size, (0, 0, 0))
        rgb_image.paste(image, mask=image.split()[3])  # 使用alpha通道作为遮罩
        
        # 转换为ComfyUI格式
        img_tensor = torch.from_numpy(np.array(rgb_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        
        return (img_tensor, mask_tensor)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "YCTextImageGenerator": TextImageGenerator
}

# 显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "YCTextImageGenerator": "文本图像生成器"
} 
