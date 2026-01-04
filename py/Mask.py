import torch
import numpy as np
import cv2

# 定义最大分辨率常量
MAX_RESOLUTION = 8192

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
                    "max": 100,
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
        # 保存原始mask和设备信息
        original_mask = mask
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
                contour_mask = np.zeros_like(mask_np)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                contour_masks[i] = contour_mask

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
            if isinstance(original_mask, torch.Tensor):
                return (original_mask,)
            else:
                mask_tensor = torch.from_numpy(original_mask).float()
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

class YCRemapMaskRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "remap"
    CATEGORY = "YCNode/Mask"
    DESCRIPTION = """
Sets new min and max values for the mask.
"""

    def remap(self, mask, min, max):

         # Find the maximum value in the mask
        mask_max = torch.max(mask)
        
        # If the maximum mask value is zero, avoid division by zero by setting it to 1
        mask_max = mask_max if mask_max > 0 else 1
        
        # Scale the mask values to the new range defined by min and max
        # The highest pixel value in the mask will be scaled to max
        scaled_mask = (mask / mask_max) * (max - min) + min
        
        # Clamp the values to ensure they are within [0.0, 1.0]
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)
        
        return (scaled_mask, )


def get_mask_polygon(mask_np):
    """Helper function to get polygon points from mask"""
    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return polygon.squeeze()

class MaskFilterBySolidity:
    """
    一个通过保留具有最高坚实度的区域来过滤遮罩的节点。
    坚实度是衡量一个形状有多"坚实"或"凸"的指标。
    它的计算方法是：面积 / 凸包面积。
    一个完美的凸形，其坚实度为1.0。
    这对于从噪声或碎片中分离出完整、非破碎的形状很有用。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "keep_top_n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "min_area": ("INT", {"default": 1000, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filtered_mask",)
    FUNCTION = "filter_by_solidity"
    CATEGORY = "YCNode/Mask"

    def filter_by_solidity(self, mask, keep_top_n=1, min_area=100):
        mask_np = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu().numpy()
        result_masks = []

        for m in mask_np:
            # 转换为8位灰度图
            mask_8bit = (m * 255).astype(np.uint8)
            
            # 找到所有外部轮廓
            contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 过滤掉面积过小的轮廓
            valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

            # 创建一个空的黑色遮罩用于存放结果
            filtered_mask = np.zeros_like(mask_8bit)

            if valid_contours:
                # 定义坚实度计算函数
                def calculate_solidity(c):
                    area = cv2.contourArea(c)
                    # 预过滤后，此检查主要是为了安全
                    if area == 0:
                        return 0
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    if hull_area == 0:
                        return 0
                    return float(area) / hull_area

                # 计算所有轮廓的坚实度
                contour_solidities = [(c, calculate_solidity(c)) for c in valid_contours]
                
                # 按坚实度从高到低排序
                contour_solidities.sort(key=lambda x: x[1], reverse=True)
                
                # 保留前keep_top_n个坚实度最高的轮廓
                top_contours = [c for c, _ in contour_solidities[:keep_top_n]]
                
                # 在新的遮罩上绘制这些轮廓
                cv2.drawContours(filtered_mask, top_contours, -1, 255, -1)

            # 转换回0-1范围的浮点数并添加到结果列表
            result_mask = filtered_mask.astype(np.float32) / 255.0
            result_masks.append(result_mask)
        
        # 将结果堆叠并转换为torch张量
        result_tensor = torch.from_numpy(np.stack(result_masks))
        
        return (result_tensor,)

class MaskResizeToRatio:
    """
    将遮罩中的白色内容调整为指定比例的遮罩
    自动检测白色内容的边界框，然后输出指定比例的遮罩
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ratio": (["1:1", "3:4", "4:3", "9:16", "16:9", "3:2", "2:3"], {
                    "default": "1:1"
                }),
                "padding_mode": (["center", "top_left", "bottom_right"], {
                    "default": "center"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("resized_mask",)
    FUNCTION = "resize_to_ratio"
    CATEGORY = "YCNode/Mask"

    def resize_to_ratio(self, mask, ratio, padding_mode="center"):
        # 1. 处理输入mask格式
        if isinstance(mask, torch.Tensor):
            device = mask.device
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            mask_np = mask.cpu().numpy()[0]
        else:
            device = torch.device('cpu')
            mask_np = mask
        
        # 2. 转换为二值图像
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # 3. 找到白色内容的边界框
        y_coords, x_coords = np.nonzero(binary_mask)
        
        if len(x_coords) == 0:
            # 如果没有白色内容，返回原始尺寸的空白遮罩
            return (torch.from_numpy(np.zeros_like(mask_np)).float().to(device),)
        
        # 4. 计算边界框
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        content_width = max_x - min_x + 1
        content_height = max_y - min_y + 1
        
        # 5. 解析目标比例
        ratio_map = {
            "1:1": (1, 1),
            "3:4": (3, 4),
            "4:3": (4, 3),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "3:2": (3, 2),
            "2:3": (2, 3)
        }
        
        target_w_ratio, target_h_ratio = ratio_map[ratio]
        
        # 6. 计算目标矩形尺寸（以原有内容尺寸为基础）
        # 确保目标矩形能完全包含原有内容
        if content_width / content_height > target_w_ratio / target_h_ratio:
            # 内容更宽，以宽度为准
            target_width = content_width
            target_height = int(content_width * target_h_ratio / target_w_ratio)
        else:
            # 内容更高，以高度为准
            target_height = content_height
            target_width = int(content_height * target_w_ratio / target_h_ratio)
        
        # 7. 创建新的遮罩（保持原始画布尺寸）
        original_height, original_width = mask_np.shape
        new_mask = np.zeros((original_height, original_width), dtype=np.float32)
        
        # 8. 计算目标矩形的位置
        if padding_mode == "center":
            # 以原有内容为中心
            content_center_x = (min_x + max_x) // 2
            content_center_y = (min_y + max_y) // 2
            start_x = content_center_x - target_width // 2
            start_y = content_center_y - target_height // 2
        elif padding_mode == "top_left":
            # 以原有内容的左上角为基准
            start_x = min_x
            start_y = min_y
        elif padding_mode == "bottom_right":
            # 以原有内容的右下角为基准
            start_x = max_x - target_width + 1
            start_y = max_y - target_height + 1
        
        # 9. 确保不超出边界
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(original_width, start_x + target_width)
        end_y = min(original_height, start_y + target_height)
        
        # 10. 在目标矩形区域填充白色
        new_mask[start_y:end_y, start_x:end_x] = 1.0
        
        # 11. 返回结果
        result_tensor = torch.from_numpy(new_mask).float().to(device)
        if len(mask.shape) == 3:
            result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)

class YCMaskDirectionExpand:
    """
    遮罩方向性扩展/缩减节点 (XY解耦版)
    
    功能：
    - 完美实现单个方向的独立扩展/缩减，互不干扰。
    - 解决缩减时"误杀"细长物体或带动其他方向的问题。
    - 逻辑：通过分离垂直和水平操作通道，实现精准的单向控制。
    
    参数：
    - 正数：向外扩展 (Expand)
    - 负数：向内缩减 (Shrink)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "top": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "正数：向上生长 | 负数：从顶部向下裁切"
                }),
                "bottom": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "正数：向下生长 | 负数：从底部向上裁切"
                }),
                "left": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "正数：向左生长 | 负数：从左侧向右裁切"
                }),
                "right": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "正数：向右生长 | 负数：从右侧向左裁切"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process_mask"
    CATEGORY = "YCNode/Mask"
    
    def process_mask(self, mask, top, bottom, left, right):
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return (mask,)

        # --- 参数分离 ---
        # 扩展参数 (只取正值)
        exp_t, exp_b = max(0, top), max(0, bottom)
        exp_l, exp_r = max(0, left), max(0, right)
        
        # 缩减参数 (只取负值并转正)
        shr_t, shr_b = abs(min(0, top)), abs(min(0, bottom))
        shr_l, shr_r = abs(min(0, left)), abs(min(0, right))

        mask_np = mask.cpu().numpy()
        result_masks = []

        # --- 核配置 (Kernel Configuration) ---
        # 关键修改：XY方向使用独立的 1D Kernel，彻底杜绝干扰
        
        # 1. 垂直扩展 (Vertical Expand) - 核形状 (1, H)
        do_exp_v = (exp_t + exp_b) > 0
        if do_exp_v:
            k_exp_v = np.ones((exp_t + exp_b + 1, 1), dtype=np.uint8)
            # 锚点Y = 底部扩展量 (逻辑：想向上扩，核要伸到下面，锚点在底部的相对反方向，即 exp_b)
            # 修正：锚点Y = exp_b。
            # 验证：Top=10, Bottom=0 -> H=11, AnchorY=0 (Top). 
            # 锚点在顶，核向下延伸，像素检测下方(白)，变白 -> 向上生长。正确。
            anchor_exp_v = (0, exp_b) 

        # 2. 水平扩展 (Horizontal Expand) - 核形状 (W, 1)
        do_exp_h = (exp_l + exp_r) > 0
        if do_exp_h:
            k_exp_h = np.ones((1, exp_l + exp_r + 1), dtype=np.uint8)
            # 锚点X = 右侧扩展量 (逻辑同上，anchor=exp_r)
            anchor_exp_h = (exp_r, 0)

        # 3. 垂直缩减 (Vertical Shrink) - 核形状 (1, H)
        do_shr_v = (shr_t + shr_b) > 0
        if do_shr_v:
            k_shr_v = np.ones((shr_t + shr_b + 1, 1), dtype=np.uint8)
            # 锚点Y = 顶部缩减量 (逻辑：从顶缩，需检查上方，核要伸到上面，锚点在底，即 shr_t)
            # 验证：TopShrink=10 (T=10, B=0) -> H=11, AnchorY=10 (Bottom).
            # 锚点在底，核向上延伸，检测上方(黑)，变黑 -> 从顶变黑。正确。
            anchor_shr_v = (0, shr_t)

        # 4. 水平缩减 (Horizontal Shrink) - 核形状 (W, 1)
        do_shr_h = (shr_l + shr_r) > 0
        if do_shr_h:
            k_shr_h = np.ones((1, shr_l + shr_r + 1), dtype=np.uint8)
            # 锚点X = 左侧缩减量 (逻辑同上，anchor=shr_l)
            anchor_shr_h = (shr_l, 0)

        for m in mask_np:
            img = (m * 255).astype(np.uint8)
            
            # 顺序执行：先扩展后缩减 (顺序可调，通常这样最符合直觉)
            # 因为是分离的XY操作，所以不会出现"扩展X导致Y方向变圆"的情况
            
            # --- 执行扩展 ---
            if do_exp_v:
                img = cv2.dilate(img, k_exp_v, anchor=anchor_exp_v, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if do_exp_h:
                img = cv2.dilate(img, k_exp_h, anchor=anchor_exp_h, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            
            # --- 执行缩减 ---
            # 关键：这里使用的是 erode，但因为核宽度/高度为1，所以只在特定轴向生效
            # 即使是一条细线，只要其长度超过缩减量，就不会被消灭
            if do_shr_v:
                img = cv2.erode(img, k_shr_v, anchor=anchor_shr_v, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if do_shr_h:
                img = cv2.erode(img, k_shr_h, anchor=anchor_shr_h, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            
            result_masks.append(img.astype(np.float32) / 255.0)
        
        result_tensor = torch.from_numpy(np.stack(result_masks))
        return (result_tensor,)


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
        right, bottom = (min(left + source.shape[-1], output.shape[-1]), min(top + source.shape[-2], output.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)
        
        # 确保不超出 source 的实际尺寸
        visible_height = min(visible_height, source.shape[-2])
        visible_width = min(visible_width, source.shape[-1])

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = output[:, top:bottom, left:right]

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
    "MaskTopNFilter": MaskTopNFilter,
    "MaskSplitFilter": MaskSplitFilter,
    "MaskContourFillNode": MaskContourFillNode,
    "YCRemapMaskRange": YCRemapMaskRange,
    "MaskFilterBySolidity": MaskFilterBySolidity,
    "MaskResizeToRatio": MaskResizeToRatio,
    "YCMaskDirectionExpand": YCMaskDirectionExpand,
    "MaskComposite": MaskComposite,
}


# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskTopNFilter": "Mask Top-N Filter",
    "MaskSplitFilter": "Mask Split Filter",
    "MaskContourFillNode": "MaskContourFill_YC",
    "YCRemapMaskRange": "Remap Mask Range (YC)",
    "MaskFilterBySolidity": "Filter Mask By Solidity",
    "MaskResizeToRatio": "Resize Mask To Ratio (YC)",
    "YCMaskDirectionExpand": "Mask Direction Expand (YC)",
    "MaskComposite": "Mask Composite (YC)",
}
