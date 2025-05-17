import torch
import numpy as np
import cv2
from scipy import ndimage
import comfy.utils

class IrregularToEllipseMask:
    """
    将不规则遮罩（包括凹陷区域）转换为椭圆形遮罩
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "keep_region": (["all", "largest", "top", "bottom", "left", "right"], {"default": "largest"}),
                "fill_holes": (["enable", "disable"], {"default": "enable"}),
                "smooth_edges": (["enable", "disable"], {"default": "enable"}),
                "smoothing_kernel_size": ("INT", {"default": 5, "min": 3, "max": 31, "step": 2}),
                "output_mode": (["ellipse", "convex_hull", "filled_original"], {"default": "ellipse"}),
                "expand_mask": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "blur_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("ellipse_mask",)
    FUNCTION = "convert_to_ellipse"
    CATEGORY = "YCNode/Mask"

    def convert_to_ellipse(self, mask, keep_region="largest", fill_holes="enable", smooth_edges="enable", 
                           smoothing_kernel_size=5, output_mode="ellipse", expand_mask=0, blur_amount=0.0):
        # 确保内核大小是奇数
        if smoothing_kernel_size % 2 == 0:
            smoothing_kernel_size += 1
            
        # 将掩码转换为numpy数组并进行处理
        mask_np = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu().numpy()
        result_masks = []
        
        for m in mask_np:
            # 转换为8位灰度图
            mask_8bit = (m * 255).astype(np.uint8)
            
            # 过滤区域（如果需要）
            if keep_region != "all":
                # 查找所有轮廓
                all_contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(all_contours) > 1:  # 只有存在多个轮廓时才进行过滤
                    filtered_mask = np.zeros_like(mask_8bit)
                    
                    if keep_region == "largest":
                        # 保留最大面积的轮廓
                        selected_contour = max(all_contours, key=cv2.contourArea)
                        cv2.drawContours(filtered_mask, [selected_contour], 0, 255, -1)
                    else:
                        # 根据位置选择轮廓
                        if keep_region == "top":
                            # 保留最上方的轮廓（y坐标最小）
                            selected_contour = min(all_contours, key=lambda c: cv2.boundingRect(c)[1])
                        elif keep_region == "bottom":
                            # 保留最下方的轮廓（y坐标最大）
                            selected_contour = max(all_contours, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3])
                        elif keep_region == "left":
                            # 保留最左方的轮廓（x坐标最小）
                            selected_contour = min(all_contours, key=lambda c: cv2.boundingRect(c)[0])
                        elif keep_region == "right":
                            # 保留最右方的轮廓（x坐标最大）
                            selected_contour = max(all_contours, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2])
                        
                        cv2.drawContours(filtered_mask, [selected_contour], 0, 255, -1)
                    
                    # 使用过滤后的遮罩替换原始遮罩
                    mask_8bit = filtered_mask
            
            # 填充内部空洞
            if fill_holes == "enable":
                # 使用形态学操作填充空洞
                mask_filled = ndimage.binary_fill_holes(mask_8bit > 127).astype(np.uint8) * 255
            else:
                mask_filled = mask_8bit
                
            # 查找轮廓
            contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建输出掩码
            result_mask = np.zeros_like(mask_filled)
            
            # 如果找到轮廓
            if contours and len(contours) > 0:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                if output_mode == "ellipse":
                    # 计算最佳拟合椭圆
                    if len(largest_contour) >= 5:  # 椭圆拟合需要至少5个点
                        ellipse = cv2.fitEllipse(largest_contour)
                        # 绘制椭圆
                        result_mask = cv2.ellipse(result_mask, ellipse, 255, -1)
                    else:
                        # 如果点太少，退化为矩形
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        result_mask = cv2.rectangle(result_mask, (x, y), (x+w, y+h), 255, -1)
                
                elif output_mode == "convex_hull":
                    # 使用凸包
                    hull = cv2.convexHull(largest_contour)
                    cv2.drawContours(result_mask, [hull], 0, 255, -1)
                
                elif output_mode == "filled_original":
                    # 直接绘制填充后的轮廓
                    cv2.drawContours(result_mask, [largest_contour], 0, 255, -1)
            
            # 应用遮罩扩展/收缩
            if expand_mask != 0:
                # 创建适当大小的内核
                kernel_size = abs(expand_mask) * 2 + 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                if expand_mask > 0:
                    # 扩展遮罩
                    result_mask = cv2.dilate(result_mask, kernel, iterations=1)
                else:
                    # 收缩遮罩
                    result_mask = cv2.erode(result_mask, kernel, iterations=1)
            
            # 平滑边缘（如果启用）
            if smooth_edges == "enable" and smoothing_kernel_size > 1:
                kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
                result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)
                result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
                result_mask = cv2.GaussianBlur(result_mask, (smoothing_kernel_size, smoothing_kernel_size), 0)
                
            # 应用额外的高斯模糊（如果指定）
            if blur_amount > 0:
                # 确保模糊内核大小是奇数
                blur_kernel_size = max(3, int(blur_amount * 2) * 2 + 1)
                result_mask = cv2.GaussianBlur(result_mask, (blur_kernel_size, blur_kernel_size), blur_amount)
            
            # 转换回0-1范围的浮点数
            result_mask = result_mask.astype(np.float32) / 255.0
            result_masks.append(result_mask)
        
        # 转换回torch张量
        result_tensor = torch.from_numpy(np.stack(result_masks))
        
        return (result_tensor,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "IrregularToEllipseMask": IrregularToEllipseMask
}

# 显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "IrregularToEllipseMask": "Irregular To EllipseMask"
} 
