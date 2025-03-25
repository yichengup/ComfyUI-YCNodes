import cv2
import numpy as np
import torch
import folder_paths
import random
from nodes import SaveImage

# 节点类定义
class MaskTopNFilter:
    def __init__(self):
        self.type = "MaskTopNFilter"
        self.output_node = True
        self.input_node = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "keep_top_n": ("INT", {
                    "default": 2, 
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("MASK",)  # 只返回过滤后的mask
    RETURN_NAMES = ("filtered_mask",)
    FUNCTION = "filter_mask"
    CATEGORY = "YCNode/Mask"

    def filter_mask(self, mask, keep_top_n):
        # 1. 处理输入mask
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 2. 自动处理通道
        if len(mask.shape) == 3:
            mask = mask[0]  # 如果是3D，取第一个通道
        elif len(mask.shape) == 4:
            mask = mask[0, 0]  # 如果是4D，取第一个batch的第一个通道
        
        # 3. 转换为二值图像
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 4. 连通区域分析
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        # 5. 处理结果
        if num_labels - 1 <= keep_top_n:
            return (torch.from_numpy(binary_mask.astype(np.float32)),)
        
        # 6. 面积计算和排序
        areas = [(i, np.sum(labels == i)) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        # 7. 生成新mask
        new_mask = np.zeros_like(binary_mask)
        for i in range(min(keep_top_n, len(areas))):
            label_idx = areas[i][0]
            new_mask[labels == label_idx] = 1
        
        return (torch.from_numpy(new_mask.astype(np.float32)),)  # 只返回mask

class MaskSplitFilter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("segmented_masks",)
    FUNCTION = "segment_mask"
    CATEGORY = "YCNode/Mask"

    def find_top_left_point(self, mask_np):
        """找到mask中最左上角的点"""
        y_coords, x_coords = np.nonzero(mask_np)
        if len(x_coords) == 0:
            return float('inf'), float('inf')
        
        min_x = np.min(x_coords)
        min_y = np.min(y_coords[x_coords == min_x])
        
        return min_x, min_y

    def segment_mask(self, mask):
        # 保存原始设备信息
        device = mask.device if isinstance(mask, torch.Tensor) else torch.device('cpu')
        
        # 确保mask是正确的形状并转换为numpy数组
        if isinstance(mask, torch.Tensor):
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            mask_np = (mask[0] * 255).cpu().numpy().astype(np.uint8)
        else:
            mask_np = (mask * 255).astype(np.uint8)
        
        # 使用OpenCV找到轮廓
        contours, hierarchy = cv2.findContours(
            mask_np, 
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask_info = []  # 用于排序的信息列表
        
        if hierarchy is not None and len(contours) > 0:
            hierarchy = hierarchy[0]
            contour_masks = {}
            
            # 创建每个轮廓的mask
            for i, contour in enumerate(contours):
                mask = np.zeros_like(mask_np)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                contour_masks[i] = mask

            # 处理每个轮廓
            processed_indices = set()
            
            for i, (contour, h) in enumerate(zip(contours, hierarchy)):
                if i in processed_indices:
                    continue
                    
                current_mask = contour_masks[i].copy()
                child_idx = h[2]
                
                if child_idx != -1:
                    while child_idx != -1:
                        current_mask = cv2.subtract(current_mask, contour_masks[child_idx])
                        processed_indices.add(child_idx)
                        child_idx = hierarchy[child_idx][0]
                
                # 找到最左上角的点用于排序
                min_x, min_y = self.find_top_left_point(current_mask)
                
                # 转换为tensor
                mask_tensor = torch.from_numpy(current_mask).float() / 255.0
                mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor = mask_tensor.to(device)
                
                mask_info.append((mask_tensor, min_x, min_y))
                processed_indices.add(i)
        
        # 如果没有找到任何轮廓，返回原始mask
        if not mask_info:
            if isinstance(mask, torch.Tensor):
                return (mask,)
            else:
                mask_tensor = torch.from_numpy(mask).float()
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor = mask_tensor.to(device)
                return (mask_tensor,)
        
        # 根据最左上角点排序
        mask_info.sort(key=lambda x: (x[1], x[2]))
        
        # 合并所有mask
        result_masks = None
        for mask_tensor, _, _ in mask_info:
            if result_masks is None:
                result_masks = mask_tensor
            else:
                result_masks = torch.cat([result_masks, mask_tensor], dim=0)
        
        return (result_masks,)

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

class MaskContourFillNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_area": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 10000,
                    "step": 10
                }),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filled_mask",)
    FUNCTION = "fill_mask_contours"
    CATEGORY = "YCNode/Mask"

    def fill_mask_contours(self, mask, min_area):
        # 保存原始设备信息
        device = mask.device if isinstance(mask, torch.Tensor) else torch.device('cpu')
        
        # 确保mask是正确的格式
        if isinstance(mask, torch.Tensor):
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            # 如果是批次多个遮罩，只取第一个
            if len(mask.shape) > 3:
                mask = mask[0].unsqueeze(0)
                
            # 转为numpy进行处理
            mask_np = mask.cpu().numpy()[0]  # [H, W]
        else:
            mask_np = mask
            
        # 确保值范围在[0, 1]并转换为8位图像
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_8bit = (mask_np * 255).astype(np.uint8)
        
        # 创建初始的填充遮罩（与输入相同）
        filled_mask = mask_8bit.copy()
        
        # 使用所有轮廓模式查找轮廓
        contours, hierarchy = cv2.findContours(mask_8bit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 填充轮廓
        if contours:
            # 创建新的全零遮罩
            filled_mask = np.zeros_like(mask_8bit)
            
            # 根据面积过滤轮廓
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
            
            # 填充通过过滤的轮廓
            for i, contour in enumerate(filtered_contours):
                # 填充遮罩
                cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # 将遮罩转回torch tensor
        filled_tensor = torch.from_numpy(filled_mask.astype(np.float32) / 255.0)
        filled_tensor = filled_tensor.unsqueeze(0)  # 添加通道维度 [1, H, W]
        filled_tensor = filled_tensor.to(device)
        
        return (filled_tensor,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MaskTopNFilter": MaskTopNFilter,
    "MaskSplitFilter": MaskSplitFilter,
    "MaskPreviewNode": MaskPreviewNode,
    "MaskContourFillNode": MaskContourFillNode
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskTopNFilter": "Mask Top-N Filter",
    "MaskSplitFilter": "Mask Split Filter",
    "MaskPreviewNode": "MaskPreview_YC",
    "MaskContourFillNode": "MaskContourFill_YC"
}
