import torch
import numpy as np
from PIL import Image

class YCConstrainImage:
    """
    A node that constrains an image to a maximum and minimum size while maintaining aspect ratio.
    YC version - Modified from original implementation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "max_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_if_required": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_image"
    CATEGORY = "YCNode/Image"
    OUTPUT_IS_LIST = (True,)

    def constrain_image(self, images, max_width, max_height, min_width, min_height, crop_if_required):
        crop_if_required = crop_if_required == "yes"
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

            current_width, current_height = img.size
            
            # 防止除零错误
            if current_height == 0 or current_width == 0:
                results.append(image)
                continue
                
            aspect_ratio = current_width / current_height

            # 修复 min 尺寸逻辑:先根据 max 限制计算,再检查是否满足 min
            if current_width / current_height > max_width / max_height:
                # 宽度先达到上限
                constrained_width = min(current_width, max_width)
                constrained_height = int(constrained_width / aspect_ratio)
            else:
                # 高度先达到上限
                constrained_height = min(current_height, max_height)
                constrained_width = int(constrained_height * aspect_ratio)

            # 如果需要裁剪且尺寸小于最小限制,放大到满足最小尺寸
            if crop_if_required:
                if constrained_width < min_width:
                    constrained_width = min_width
                    constrained_height = int(constrained_width / aspect_ratio)
                if constrained_height < min_height:
                    constrained_height = min_height
                    constrained_width = int(constrained_height * aspect_ratio)

            # 确保最终尺寸为正数
            constrained_width = max(constrained_width, 1)
            constrained_height = max(constrained_height, 1)

            resized_image = img.resize((constrained_width, constrained_height), Image.LANCZOS)

            # 如果开启裁剪且超出最大尺寸,居中裁剪
            if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
                left = max((constrained_width - max_width) // 2, 0)
                top = max((constrained_height - max_height) // 2, 0)
                right = min(constrained_width, max_width + left)
                bottom = min(constrained_height, max_height + top)
                resized_image = resized_image.crop((left, top, right, bottom))

            resized_image = np.array(resized_image).astype(np.float32) / 255.0
            resized_image = torch.from_numpy(resized_image)[None,]
            results.append(resized_image)
                
        return (results,)

NODE_CLASS_MAPPINGS = {
    "YCConstrainImage": YCConstrainImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCConstrainImage": "YC Constrain Image",
}
