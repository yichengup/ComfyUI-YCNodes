import torch
import numpy as np
import cv2

def resize_with_aspect_ratio(img, target_size, target_dim='width', interpolation=cv2.INTER_CUBIC):
    """等比例缩放图片"""
    h, w = img.shape[:2]
    if target_dim == 'width':
        aspect = h / w
        new_w = target_size
        new_h = int(aspect * new_w)
    else:
        aspect = w / h
        new_h = target_size
        new_w = int(aspect * new_h)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

def find_content_bounds(mask):
    """找到内容的边界"""
    if not mask.any():  # 如果遮罩全为0
        return (0, 0, mask.shape[1], mask.shape[0])
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max + 1, y_max + 1

def safe_divide(a, b, default=1):
    """安全除法，避免除以0"""
    return a / b if b != 0 else default

def screen_blend(mask1, mask2):
    """滤色模式混合两个遮罩"""
    return 255 - ((255 - mask1.astype(float)) * (255 - mask2.astype(float)) / 255)

class ImageICAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "first_image": ("IMAGE",),
                "second_image": ("IMAGE",),
                "reference_edge": (["image1_width", "image1_height", "image2_width", "image2_height"], {
                    "default": "image1_width",
                }),
                "combine_mode": (["horizontal", "vertical", "overlay"], {
                    "default": "horizontal",
                }),
                "second_image_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "position_type": (["top", "center", "bottom", "left", "right"], {
                    "default": "center",
                }),
                "x_position": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "y_position": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "final_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                }),
            },
            "optional": {
                "first_mask": ("MASK",),
                "second_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "IMAGE", "TUPLE", "TUPLE")
    RETURN_NAMES = ("IMAGE", "MASK", "FIRST_MASK", "SECOND_MASK", "MAIN_IMAGE", "first_size", "second_size")
    FUNCTION = "combine_images"
    CATEGORY = "YiCheng/Image"

    def combine_images(self, first_image, second_image, reference_edge, combine_mode, 
                      second_image_scale, position_type, x_position, y_position, final_size, background_color,
                      first_mask=None, second_mask=None):
        # 获取输入图像并确保数据类型正确
        image1 = (first_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        image2 = (second_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        
        # 获取原始尺寸
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # 确定主图和次图
        is_first_main = reference_edge.startswith('image1')
        main_image = image1 if is_first_main else image2
        second_img = image2 if is_first_main else image1
        main_h, main_w = main_image.shape[:2]
        
        # 验证并处理第一个遮罩
        if first_mask is not None:
            mask1_h, mask1_w = first_mask[0].shape
            if mask1_h != h1 or mask1_w != w1:
                # 如果尺寸不匹配，创建新的空遮罩
                first_mask = np.zeros((h1, w1), dtype=np.float32)
            else:
                first_mask = first_mask[0].numpy()
        else:
            first_mask = np.zeros((h1, w1), dtype=np.float32)

        # 验证并处理第二个遮罩
        if second_mask is not None:
            mask2_h, mask2_w = second_mask[0].shape
            if mask2_h != h2 or mask2_w != w2:
                # 如果尺寸不匹配，创建新的空遮罩
                second_mask = np.zeros((h2, w2), dtype=np.float32)
            else:
                second_mask = second_mask[0].numpy()
        else:
            second_mask = np.zeros((h2, w2), dtype=np.float32)

        # 确定主遮罩和次遮罩
        main_mask = first_mask if is_first_main else second_mask
        second_mask = second_mask if is_first_main else first_mask

        # 将遮罩转换为0-255范围用于处理
        first_mask_255 = (first_mask * 255).astype(np.uint8)
        second_mask_255 = (second_mask * 255).astype(np.uint8)

        # 转换背景颜色
        if background_color.startswith('#'):
            bg_color = tuple(int(background_color[i:i+2], 16) for i in (5, 3, 1))[::-1]

        # 根据基准边计算目标尺寸
        target_size = main_w if reference_edge.endswith('width') else main_h
        target_dim = 'width' if reference_edge.endswith('width') else 'height'

        # 等比例缩放第二张图片
        scaled_second = resize_with_aspect_ratio(second_img, target_size, target_dim)
        scaled_second_mask = resize_with_aspect_ratio(second_mask, target_size, target_dim, cv2.INTER_LINEAR)

        # 第二张图片额外缩放
        if second_image_scale != 1.0:
            h, w = scaled_second.shape[:2]
            new_w = int(w * second_image_scale)
            new_h = int(h * second_image_scale)
            scaled_second = cv2.resize(scaled_second, (new_w, new_h))
            scaled_second_mask = cv2.resize(scaled_second_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 将缩放后的遮罩转换为255范围
        scaled_second_mask_255 = (scaled_second_mask * 255).astype(np.uint8)

        # 创建画布
        if combine_mode == "overlay":
            canvas_w = main_w + scaled_second.shape[1]
            canvas_h = max(main_h, scaled_second.shape[0])
        elif combine_mode == "horizontal":
            canvas_w = main_w + scaled_second.shape[1]
            canvas_h = max(main_h, scaled_second.shape[0])
        else:  # vertical
            canvas_w = max(main_w, scaled_second.shape[1])
            canvas_h = main_h + scaled_second.shape[0]

        # 创建画布和遮罩
        final_canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
        final_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        first_separate_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        second_separate_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # 放置主图（总是在左边或顶部）
        y1 = (canvas_h - main_h) // 2 if combine_mode == "horizontal" else 0
        x1 = 0
        final_canvas[y1:y1+main_h, x1:x1+main_w] = main_image

        # 创建主图区域遮罩
        main_region_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        main_region_mask[y1:y1+main_h, x1:x1+main_w] = 255
        if is_first_main:
            first_separate_mask = main_region_mask.copy()
            final_mask[y1:y1+main_h, x1:x1+main_w] = first_mask_255
        else:
            second_separate_mask = main_region_mask.copy()
            final_mask[y1:y1+main_h, x1:x1+main_w] = second_mask_255

        # 放置第二张图片
        h2, w2 = scaled_second.shape[:2]
        if combine_mode == "overlay":
            # 使用百分比位置
            x2 = int((canvas_w - w2) * x_position / 100)
            y2 = int((canvas_h - h2) * y_position / 100)
        elif combine_mode == "horizontal":
            x2 = main_w if position_type != "left" else 0
            if position_type == "top":
                y2 = 0
            elif position_type == "bottom":
                y2 = canvas_h - h2
            else:  # center
                y2 = (canvas_h - h2) // 2
        else:  # vertical
            if position_type == "left":
                x2 = 0
            elif position_type == "right":
                x2 = canvas_w - w2
            else:  # center
                x2 = (canvas_w - w2) // 2
            y2 = main_h

        # 确保坐标不会超出画布范围
        x2 = max(0, min(x2, canvas_w - w2))
        y2 = max(0, min(y2, canvas_h - h2))

        # 将第二张图片放入画布
        final_canvas[y2:y2+h2, x2:x2+w2] = scaled_second

        # 创建第二张图区域遮罩
        second_region_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        second_region_mask[y2:y2+h2, x2:x2+w2] = 255
        if is_first_main:
            second_separate_mask = second_region_mask.copy()
            # 使用滤色模式合并遮罩
            second_mask_area = np.zeros_like(final_mask)
            second_mask_area[y2:y2+h2, x2:x2+w2] = scaled_second_mask_255
            final_mask = screen_blend(final_mask, second_mask_area).astype(np.uint8)
        else:
            first_separate_mask = second_region_mask.copy()
            # 使用滤色模式合并遮罩
            first_mask_area = np.zeros_like(final_mask)
            first_mask_area[y2:y2+h2, x2:x2+w2] = scaled_second_mask_255
            final_mask = screen_blend(final_mask, first_mask_area).astype(np.uint8)

        # 找到有效内容区域
        content_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        content_mask[final_mask > 0] = 255  # 遮罩区域
        content_mask[np.any(final_canvas != bg_color, axis=2)] = 255  # 非背景色区域
        
        # 获取有效区域边界
        x_min, y_min, x_max, y_max = find_content_bounds(content_mask)
        
        # 确保边界有效
        x_min = min(x_min, canvas_w - 1)
        y_min = min(y_min, canvas_h - 1)
        x_max = max(x_min + 1, min(x_max, canvas_w))
        y_max = max(y_min + 1, min(y_max, canvas_h))
        
        # 裁剪到有效区域
        final_canvas = final_canvas[y_min:y_max, x_min:x_max]
        final_mask = final_mask[y_min:y_max, x_min:x_max]
        first_separate_mask = first_separate_mask[y_min:y_max, x_min:x_max]
        second_separate_mask = second_separate_mask[y_min:y_max, x_min:x_max]

        # 计算最终尺寸
        h, w = final_canvas.shape[:2]
        if w > h:
            aspect = safe_divide(h, w)
            new_w = final_size
            new_h = int(aspect * new_w)
        else:
            aspect = safe_divide(w, h)
            new_h = final_size
            new_w = int(aspect * new_h)

        # 确保最小尺寸
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # 计算缩放比例
        scale_x = safe_divide(new_w, w)
        scale_y = safe_divide(new_h, h)

        # 计算主图在最终画布中的位置和尺寸，保持原始比例
        main_aspect = main_h / main_w
        if main_w > main_h:
            main_final_w = min(new_w, int(new_h / main_aspect))
            main_final_h = int(main_final_w * main_aspect)
        else:
            main_final_h = min(new_h, int(new_w * main_aspect))
            main_final_w = int(main_final_h / main_aspect)

        # 计算主图位置，保持在画布内
        main_x = int((x1 - x_min) * scale_x)
        main_y = int((y1 - y_min) * scale_y)
        
        # 确保主图位置有效
        main_x = min(max(0, main_x), new_w - main_final_w)
        main_y = min(max(0, main_y), new_h - main_final_h)

        # 准备主图输出（保持原始比例）
        if main_final_w > 0 and main_final_h > 0:
            main_region = cv2.resize(main_image, (main_final_w, main_final_h), interpolation=cv2.INTER_LANCZOS4)
            main_output = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)
            main_output[main_y:main_y+main_final_h, main_x:main_x+main_final_w] = main_region
        else:
            main_output = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)

        # 调整所有图像的大小
        final_canvas = cv2.resize(final_canvas, (new_w, new_h))
        final_mask = cv2.resize(final_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        first_separate_mask = cv2.resize(first_separate_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        second_separate_mask = cv2.resize(second_separate_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 转换为tensor格式
        final_canvas = final_canvas.astype(np.float32) / 255.0
        final_canvas = torch.from_numpy(final_canvas)[None,]
        
        # 将遮罩转换为float32类型，保持0-1范围
        final_mask = final_mask.astype(np.float32) / 255.0
        first_separate_mask = first_separate_mask.astype(np.float32) / 255.0
        second_separate_mask = second_separate_mask.astype(np.float32) / 255.0
        
        # 转换为tensor
        final_mask = torch.from_numpy(final_mask)[None,]
        first_separate_mask = torch.from_numpy(first_separate_mask)[None,]
        second_separate_mask = torch.from_numpy(second_separate_mask)[None,]
        main_output = torch.from_numpy(main_output.astype(np.float32) / 255.0)[None,]

        # 返回原始尺寸信息
        first_size = (w1, h1)
        second_size = (w2, h2)

        return (final_canvas, final_mask, first_separate_mask, second_separate_mask, main_output, first_size, second_size)

NODE_CLASS_MAPPINGS = {
    "ImageICAdvanced": ImageICAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageICAdvanced": "Image IC Advanced"
} 
