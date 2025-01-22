import torch
import numpy as np

class MaskFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "batch_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 63,
                    "step": 1
                }),
                "length": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 64,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("selected_masks",)
    FUNCTION = "batch_select"
    CATEGORY = "YiCheng/Mask/Batch"

    def batch_select(self, masks, batch_index, length):
        # 确保mask是正确的形状
        if isinstance(masks, torch.Tensor):
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(0)
        
        # 获取批次大小
        batch_size = masks.shape[0]
        
        # 确保索引和长度在有效范围内
        batch_index = min(batch_size - 1, batch_index)
        length = min(batch_size - batch_index, length)
        
        # 选择指定的批次
        selected_masks = masks[batch_index:batch_index + length].clone()
        
        return (selected_masks,)

class MaskRepeatBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "amount": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 64,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("repeated_masks",)
    FUNCTION = "repeat"
    CATEGORY = "YiCheng/Mask/Batch"

    def repeat(self, masks, amount):
        # 确保mask是正确的形状
        if isinstance(masks, torch.Tensor):
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(0)
        
        # 重复指定次数
        repeated_masks = masks.repeat((amount, 1, 1))
        
        return (repeated_masks,)

class MaskBatchCopy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("masks_1", "masks_2")
    FUNCTION = "copy"
    CATEGORY = "YiCheng/Mask/Batch"

    def copy(self, masks):
        # 确保mask是正确的形状
        if isinstance(masks, torch.Tensor):
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(0)
        
        # 创建两个独立的副本
        copy1 = masks.clone()
        copy2 = masks.clone()
        
        return (copy1, copy2)

class MaskBatchComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks_1": ("MASK",),
                "masks_2": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("combined_masks",)
    FUNCTION = "combine"
    CATEGORY = "YiCheng/Mask/Batch"

    def combine(self, masks_1, masks_2):
        # 确保两个mask都是正确的形状
        if isinstance(masks_1, torch.Tensor):
            if len(masks_1.shape) == 2:
                masks_1 = masks_1.unsqueeze(0)
        if isinstance(masks_2, torch.Tensor):
            if len(masks_2.shape) == 2:
                masks_2 = masks_2.unsqueeze(0)
        
        # 合并mask批次
        combined_masks = torch.cat([masks_1, masks_2], dim=0)
        
        return (combined_masks,)

# 更新节点注册
NODE_CLASS_MAPPINGS = {
    "MaskFromBatch": MaskFromBatch,
    "MaskRepeatBatch": MaskRepeatBatch,
    "MaskBatchCopy": MaskBatchCopy,
    "MaskBatchComposite": MaskBatchComposite
}

# 更新显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromBatch": "Mask From Batch",
    "MaskRepeatBatch": "Mask Repeat Batch",
    "MaskBatchCopy": "Mask Batch Copy",
    "MaskBatchComposite": "Mask Batch Composite"
} 