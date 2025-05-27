import torch
import numpy as np

# 定义最大分辨率常量
MAX_RESOLUTION = 8192

class MaskComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("MASK",),
                "source": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "operation": (["multiply", "add", "subtract", "and", "or", "xor", "overlay", "natural_blend"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_power": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "调整自然混合的强度，值越高越接近简单相加"}),
            }
        }

    CATEGORY = "YCNode/Mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"

    def combine(self, destination, source, x, y, operation, opacity=1.0, blend_power=0.5):
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))

        left, top = (x, y,)
        right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = destination[:, top:bottom, left:right]

        # 应用不透明度
        if opacity < 1.0:
            source_portion = source_portion * opacity

        if operation == "multiply":
            output[:, top:bottom, left:right] = destination_portion * source_portion
        elif operation == "add":
            output[:, top:bottom, left:right] = destination_portion + source_portion
        elif operation == "subtract":
            output[:, top:bottom, left:right] = destination_portion - source_portion
        elif operation == "and":
            output[:, top:bottom, left:right] = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "or":
            output[:, top:bottom, left:right] = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "xor":
            output[:, top:bottom, left:right] = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "overlay":
            # 实现Photoshop的overlay模式
            # 当底图<0.5时，结果=2*底图*叠加图；当底图>=0.5时，结果=1-2*(1-底图)*(1-叠加图)
            low_mask = destination_portion < 0.5
            high_mask = ~low_mask
            result = torch.zeros_like(destination_portion)
            result[low_mask] = 2 * destination_portion[low_mask] * source_portion[low_mask]
            result[high_mask] = 1 - 2 * (1 - destination_portion[high_mask]) * (1 - source_portion[high_mask])
            output[:, top:bottom, left:right] = result
        elif operation == "natural_blend":
            # 自然混合模式 - 专为遮罩的自然过渡设计
            # 在暗部使用screen模式，在亮部使用加权平均
            # 公式: dest + src - dest*src (类似screen) 加上 权重控制
            
            # 基础混合（类似screen模式，但保留更多亮度信息）
            base_blend = destination_portion + source_portion - destination_portion * source_portion
            
            # 加权求和（在白色区域避免过亮）
            sum_weighted = torch.clamp(destination_portion + source_portion, 0.0, 1.0)
            
            # 计算混合因子，决定使用多少screen效果和多少加权平均
            # 当像素越亮时(接近1)，越倾向于使用加权平均而非screen
            bright_areas = (destination_portion + source_portion) / 2.0
            blend_factor = torch.pow(bright_areas, 2.0 - blend_power)  # 可调整的指数，控制过渡点
            
            # 根据混合因子在两种模式间平滑过渡
            result = base_blend * (1.0 - blend_factor) + sum_weighted * blend_factor
            
            output[:, top:bottom, left:right] = result

        output = torch.clamp(output, 0.0, 1.0)

        return (output,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YCMaskComposite": MaskComposite
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "YCMaskComposite": "Mask Composite (YC)"
}
