import numpy as np
import cv2
import torch

class MaskSmartValleySplit:
    """智能遮罩分割节点,基于凹陷检测和模式分析"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "valley_depth_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "pattern_check_range": ("INT", {"default": 10, "min": 5, "max": 50, "step": 1}),
                "cut_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "smart_split"
    CATEGORY = "YCNode/Mask"

    def check_connectivity(self, mask):
        """检查mask是否已经分离"""
        mask_np = mask.cpu().numpy()
        # 确保mask是2D
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        num_labels, _ = cv2.connectedComponents(mask_np.astype(np.uint8))
        return num_labels > 2

    def find_major_regions(self, mask_np):
        """找到两个主要白色区域的范围"""
        # 确保mask是2D
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
            
        # 计算每列的像素和
        col_sums = np.sum(mask_np, axis=0)
        
        # 使用阈值找到显著的白色区域
        threshold = np.max(col_sums) * 0.3
        significant_cols = col_sums > threshold
        
        # 找到连续的白色区域
        regions = []
        start = None
        for i, is_significant in enumerate(significant_cols):
            if is_significant and start is None:
                start = i
            elif not is_significant and start is not None:
                regions.append((start, i))
                start = None
        
        if start is not None:
            regions.append((start, len(significant_cols)))
        
        # 按区域大小排序，返回最大的两个区域
        if not regions:
            return None
            
        regions.sort(key=lambda x: x[1] - x[0], reverse=True)
        if len(regions) < 2:
            return None
            
        # 确保两个区域之间有足够的距离
        region1, region2 = regions[:2]
        if abs(region1[1] - region2[0]) < 2:  # 确保有至少2个像素的间隔
            return None
            
        return [region1, region2]

    def find_largest_valley(self, col_sums):
        """查找最大的凹陷区域"""
        # 使用滑动窗口平滑处理，减少噪声影响
        window_size = 5
        smoothed_sums = np.convolve(col_sums, np.ones(window_size)/window_size, mode='valid')
        
        # 计算全局特征
        global_max = np.max(smoothed_sums)
        global_min = np.min(smoothed_sums)
        global_mean = np.mean(smoothed_sums)
        
        # 找到所有局部峰值
        peaks = []
        for i in range(1, len(smoothed_sums)-1):
            if smoothed_sums[i] > smoothed_sums[i-1] and smoothed_sums[i] > smoothed_sums[i+1]:
                # 只要高于平均值就考虑
                if smoothed_sums[i] > global_mean * 0.5:  # 进一步降低阈值
                    peaks.append((i, smoothed_sums[i]))
        
        if len(peaks) < 2:
            return None
            
        # 对峰值按高度排序
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # 找到最显著的凹陷
        max_valley_score = 0
        max_valley_range = None
        
        # 分析每对峰值之间的区域
        for i in range(len(peaks)):
            for j in range(i+1, len(peaks)):
                left_peak = peaks[i]
                right_peak = peaks[j]
                
                # 获取两峰之间的区域
                start_idx = min(left_peak[0], right_peak[0])
                end_idx = max(left_peak[0], right_peak[0])
                
                # 分析区域
                region = smoothed_sums[start_idx:end_idx]
                if len(region) == 0:  # 只检查是否为空
                    continue
                    
                region_min = np.min(region)
                region_min_idx = start_idx + np.argmin(region)
                
                # 计算凹陷特征
                # 1. 深度分数：凹陷的深度
                peak_height = min(left_peak[1], right_peak[1])
                depth = peak_height - region_min
                depth_score = depth / global_max
                
                # 2. 对比分数：与周围区域的对比度
                contrast = (left_peak[1] + right_peak[1])/2 - region_min
                contrast_score = contrast / global_max
                
                # 简化评分机制，主要关注深度和对比度
                valley_score = depth_score * 3.0 + contrast_score * 2.0
                
                # 额外奖励：如果凹陷明显
                if depth_score > 0.15:  # 进一步降低深度阈值
                    valley_score *= 1.5
                
                if valley_score > max_valley_score:
                    max_valley_score = valley_score
                    max_valley_range = (start_idx, end_idx)
        
        return max_valley_range

    def detect_valley(self, mask, threshold):
        """检测两个主要区域之间的凹陷"""
        mask_np = mask.cpu().numpy()
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
            
        # 首先尝试找到主要区域
        major_regions = self.find_major_regions(mask_np)
        if major_regions:
            # 按照x坐标排序区域
            major_regions.sort(key=lambda x: x[0])
            valley_start = major_regions[0][1]  # 第一个区域的结束
            valley_end = major_regions[1][0]    # 第二个区域的开始
            
            # 确保凹陷区域有效
            if valley_end > valley_start and valley_end - valley_start >= 2:
                return valley_start, valley_end
        
        # 如果主要区域检测失败，尝试查找最大凹陷
        col_sums = np.sum(mask_np, axis=0)
        valley_range = self.find_largest_valley(col_sums)
        
        return valley_range

    def analyze_pattern(self, mask, valley_range, check_range):
        """分析凹陷区域的整体趋势"""
        if valley_range is None:
            return False, None
        
        start, end = valley_range
        if end <= start or end - start < 2:
            return False, None
            
        mask_np = mask.cpu().numpy()
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        # 获取凹陷区域的像素分布
        col_sums = np.sum(mask_np[:, start:end], axis=0)
        if len(col_sums) < 3:  # 确保有足够的数据进行分析
            return False, None
        
        # 使用平滑处理减少噪声
        window_size = max(3, min((end - start) // 5, 7))  # 限制窗口大小在3-7之间
        if len(col_sums) < window_size:  # 如果数据点太少，调整窗口大小
            window_size = len(col_sums)
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 3:
                return False, start + np.argmin(col_sums)  # 直接返回最小值位置
        
        smoothed_sums = np.convolve(col_sums, np.ones(window_size)/window_size, mode='valid')
        if len(smoothed_sums) < 3:  # 确保平滑后有足够的数据
            return False, start + np.argmin(col_sums)
        
        # 分析整体趋势
        third = max(1, len(smoothed_sums) // 3)
        first_third = smoothed_sums[:third]
        middle_third = smoothed_sums[third:2*third]
        last_third = smoothed_sums[2*third:]
        
        # 确保所有部分都有数据
        if len(first_third) == 0 or len(middle_third) == 0 or len(last_third) == 0:
            return False, start + np.argmin(col_sums)
        
        # 判断是否符合低高低模式
        is_valley_pattern = (np.mean(middle_third) > np.mean(first_third) and 
                           np.mean(middle_third) > np.mean(last_third))
        
        if is_valley_pattern:
            # 在中间区域的中心切割
            cut_pos = start + len(col_sums) // 2
        else:
            # 在最窄处切割
            cut_pos = start + np.argmin(col_sums)
            
        return is_valley_pattern, cut_pos

    def execute_cut(self, mask, cut_position, cut_width):
        """执行切割"""
        if cut_position is None:
            return mask
            
        result = mask.clone()
        half_width = cut_width // 2
        cut_start = max(0, cut_position - half_width)
        cut_end = min(mask.shape[1], cut_position + half_width + 1)
        result[:, cut_start:cut_end] = 0
        return result

    def smart_split(self, mask, valley_depth_threshold, pattern_check_range, cut_width):
        """主处理函数"""
        # 处理维度问题
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        
        # 确保mask是二值图像
        binary_mask = (mask > 0).float()
        
        # 检查是否已经分离
        if self.check_connectivity(binary_mask):
            return (binary_mask,)
            
        # 检测凹陷区域
        valley_range = self.detect_valley(binary_mask, valley_depth_threshold)
        
        # 如果没有检测到凹陷,返回原始mask
        if valley_range is None:
            return (binary_mask,)
            
        # 分析凹陷区域模式并确定切割位置
        has_pattern, cut_position = self.analyze_pattern(
            binary_mask, valley_range, pattern_check_range
        )
        
        # 执行切割
        result = self.execute_cut(binary_mask, cut_position, cut_width)
        
        # 确保返回的mask维度正确
        if len(result.shape) == 2:
            result = result.unsqueeze(0)
            
        return (result,)
        
NODE_CLASS_MAPPINGS = {
    "MaskSmartValleySplit": MaskSmartValleySplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSmartValleySplit": "Mask Smart Valley Split"
}
