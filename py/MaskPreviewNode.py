import torch
import folder_paths
import random
from nodes import SaveImage

class MaskPreviewNode(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview_mask"
    CATEGORY = "YCNode/Mask"
    OUTPUT_NODE = True
    
    def preview_mask(self, mask, prompt=None, extra_pnginfo=None):
        # 处理批处理维度和通道
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        preview = preview.movedim(1, -1).expand(-1, -1, -1, 3)
        preview = torch.clamp(preview, 0.0, 1.0)
        return self.save_images(preview, "mask_preview", prompt, extra_pnginfo)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MaskPreviewNode": MaskPreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskPreviewNode": "MaskPreview_YC"
}

