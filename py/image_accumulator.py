import torch
import numpy as np
from PIL import Image
import os
import time
import json
import hashlib
import folder_paths
from typing import List, Dict, Tuple, Optional

# yicheng author
class YCImageAccumulator:
    """
    图像累积展示节点 - 累积保存所有生成的图像，支持选择输出
    
    功能特点：
    1. 累积显示所有生成的图像（不会被清除）
    2. 支持从累积的图像中选择任意一张输出
    3. 支持不同尺寸/比例的图像
    4. 优化内存使用：完整图像保存到文件，内存中只保留元数据
    5. 自动限制最大数量，防止无限累积
    """
    
    def __init__(self):
        """初始化节点状态"""
        # 图像元数据列表（内存中只保存元数据，不保存完整图像）
        self.image_metadata: List[Dict] = []
        
        # 累积图像的保存目录（使用ComfyUI临时目录的子目录）
        temp_dir = folder_paths.get_temp_directory()
        self.gallery_dir = os.path.join(temp_dir, "image_accumulator")
        os.makedirs(self.gallery_dir, exist_ok=True)
        
        # 当前会话ID（用于区分不同的累积会话）
        self.session_id = self._generate_session_id()
        
    def _generate_session_id(self) -> str:
        """生成唯一的会话ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要累积的输入图像"}),
                "selected_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999,
                    "step": 1,
                    "tooltip": "选择要输出的图像索引。-1表示最新图像，0表示第一张"
                }),
            },
            "optional": {
                "clear_gallery": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "清空所有累积的图像"
                }),
                "max_images": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "最大累积图像数量，超过后会替换指定位置的图像"
                }),
                "update_position": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "step": 1,
                    "tooltip": "达到上限后，新图像替换的位置。-1=最后一张，0=第1张，1=第2张..."
                }),
                "batch_start": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "step": 1,
                    "tooltip": "批次输出起始索引。-1=单张输出（使用selected_index），0开始=批次输出"
                }),
                "batch_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "step": 1,
                    "tooltip": "批次输出结束索引。例如：start=0,end=2输出3张(0,1,2)"
                }),
                "align_mode": (["first", "largest", "smallest"], {
                    "default": "largest",
                    "tooltip": "批次对齐模式：first=第一张尺寸，largest=最大尺寸，smallest=最小尺寸"
                }),
                "pad_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "扩图填充颜色（十六进制，如 #000000=黑色, #FFFFFF=白色, #808080=灰色）"
                }),
                "save_to_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否将累积的图像也保存到output目录（持久化）"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_NODE = True  # 标记为输出节点，以便在UI中显示图像
    FUNCTION = "accumulate_and_select"
    CATEGORY = "YCNode/Image"
    DESCRIPTION = "累积保存所有生成的图像，支持单张或批次输出。适合对比多次生成的结果。"
    
    def accumulate_and_select(
        self, 
        image: torch.Tensor, 
        selected_index: int = -1,
        clear_gallery: bool = False,
        max_images: int = 30,
        update_position: int = -1,
        batch_start: int = -1,
        batch_end: int = -1,
        align_mode: str = "largest",
        pad_color: str = "#000000",
        save_to_output: bool = False
    ) -> Dict:
        """
        主处理函数：累积图像并选择输出
        
        Args:
            image: 输入的图像张量 [B, H, W, C]
            selected_index: 单张输出的图像索引（-1表示最新）
            clear_gallery: 是否清空图像库
            max_images: 最大累积数量
            update_position: 达到上限后替换的位置（-1表示最后一张）
            batch_start: 批次输出起始索引（-1表示单张输出）
            batch_end: 批次输出结束索引
            align_mode: 批次对齐模式（first/largest/smallest）
            pad_color: 扩图填充颜色（十六进制）
            save_to_output: 是否保存到output目录
            
        Returns:
            包含UI信息和输出结果的字典
        """
        # 1. 清空图像库（如果需要）
        if clear_gallery:
            self._clear_gallery()
            # 重新生成会话ID
            self.session_id = self._generate_session_id()
        
        # 2. 处理输入图像（可能是batch）
        batch_size = image.shape[0]
        for i in range(batch_size):
            single_image = image[i:i+1]  # 保持4维 [1, H, W, C]
            self._save_image_to_gallery(single_image, save_to_output, max_images, update_position)
        
        # 3. 准备UI展示数据
        ui_images = self._prepare_ui_images()
        
        # 4. 判断输出模式：单张 or 批次
        if batch_start == -1:
            # 单张输出模式
            output_images, output_masks = self._select_single_image(selected_index)
        else:
            # 批次输出模式
            output_images, output_masks = self._select_batch_images(batch_start, batch_end, align_mode, pad_color)
        
        # 5. 返回结果
        return {
            "ui": {
                "images": ui_images  # 前端显示所有图像
            },
            "result": (output_images, output_masks)
        }
    
    def _save_image_to_gallery(self, image: torch.Tensor, save_to_output: bool = False, max_images: int = 30, update_position: int = -1):
        """
        保存单张图像到图像库
        
        改进逻辑：
        - 如果未达到上限：正常添加新图像
        - 如果达到上限：替换指定位置的图像（默认最后一张）
        
        Args:
            image: 图像张量 [1, H, W, C]
            save_to_output: 是否同时保存到output目录
            max_images: 最大图像数量
            update_position: 达到上限后替换的位置（-1=最后一张，0=第1张，1=第2张...）
        """
        # 获取图像信息
        height, width = image.shape[1], image.shape[2]
        timestamp = time.time()
        
        # 检查是否达到上限
        is_full = len(self.image_metadata) >= max_images
        
        if is_full:
            # 达到上限：确定要替换的位置
            if update_position == -1:
                # 替换最后一张（默认）
                replace_index = len(self.image_metadata) - 1
                position_desc = "最后一张"
            elif 0 <= update_position < len(self.image_metadata):
                # 替换指定位置
                replace_index = update_position
                position_desc = f"第{update_position + 1}张"
            else:
                # 超出范围，回退到最后一张
                replace_index = len(self.image_metadata) - 1
                position_desc = "最后一张"
                print(f"[ImageAccumulator] ⚠️  update_position={update_position} 超出范围(0~{len(self.image_metadata)-1})，使用最后一张")
            
            print(f"[ImageAccumulator] 📦 图像库已满({max_images}张)，替换{position_desc}(索引{replace_index})")
            
            # 删除要替换位置的图像文件
            old_metadata = self.image_metadata[replace_index]
            if os.path.exists(old_metadata["temp_path"]):
                try:
                    os.remove(old_metadata["temp_path"])
                except Exception as e:
                    print(f"[ImageAccumulator] 删除旧文件失败: {e}")
            
            # 使用相同的索引（保持索引连续性）
            index = old_metadata["index"]
            
            # 生成新文件名（使用相同索引）
            filename = f"{self.session_id}_{index:03d}_{width}x{height}.png"
            
            # 保存新图像到临时目录
            temp_filepath = os.path.join(self.gallery_dir, filename)
            self._save_tensor_to_file(image, temp_filepath)
            
            # 可选：保存到output目录
            output_filepath = None
            if save_to_output:
                output_dir = folder_paths.get_output_directory()
                output_subdir = os.path.join(output_dir, "image_accumulator", self.session_id)
                os.makedirs(output_subdir, exist_ok=True)
                output_filepath = os.path.join(output_subdir, filename)
                self._save_tensor_to_file(image, output_filepath)
            
            # 更新指定位置的元数据
            self.image_metadata[replace_index] = {
                "index": index,
                "filename": filename,
                "temp_path": temp_filepath,
                "output_path": output_filepath,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 2),
                "timestamp": timestamp,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "size_kb": os.path.getsize(temp_filepath) / 1024 if os.path.exists(temp_filepath) else 0
            }
            
            print(f"[ImageAccumulator] ♻️  已更新{position_desc} #{index}: {width}x{height} @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
            
        else:
            # 未达到上限：正常添加新图像
            index = len(self.image_metadata)
            
            # 生成文件名
            filename = f"{self.session_id}_{index:03d}_{width}x{height}.png"
            
            # 保存到临时目录（用于UI显示和后续读取）
            temp_filepath = os.path.join(self.gallery_dir, filename)
            self._save_tensor_to_file(image, temp_filepath)
            
            # 可选：同时保存到output目录（持久化）
            output_filepath = None
            if save_to_output:
                output_dir = folder_paths.get_output_directory()
                output_subdir = os.path.join(output_dir, "image_accumulator", self.session_id)
                os.makedirs(output_subdir, exist_ok=True)
                output_filepath = os.path.join(output_subdir, filename)
                self._save_tensor_to_file(image, output_filepath)
            
            # 添加新的元数据
            metadata = {
                "index": index,
                "filename": filename,
                "temp_path": temp_filepath,
                "output_path": output_filepath,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 2),
                "timestamp": timestamp,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "size_kb": os.path.getsize(temp_filepath) / 1024 if os.path.exists(temp_filepath) else 0
            }
            
            self.image_metadata.append(metadata)
            
            print(f"[ImageAccumulator] ➕ 已累积图像 #{index}: {width}x{height} @ {metadata['time_str']}")
    
    def _save_tensor_to_file(self, image: torch.Tensor, filepath: str):
        """
        将图像张量保存为PNG文件
        
        Args:
            image: 图像张量 [1, H, W, C]，值范围 [0, 1]
            filepath: 保存路径
        """
        # 转换为numpy数组
        i = 255.0 * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 保存为PNG
        img.save(filepath, compress_level=4)
    
    def _limit_gallery_size(self, max_images: int):
        """
        限制图像库大小（已废弃，逻辑移至_save_image_to_gallery）
        
        新逻辑：
        - 在保存图像时就判断是否达到上限
        - 达到上限时替换最后一张，而不是删除最旧的
        - 这样可以保持前面的历史记录不变
        
        Args:
            max_images: 最大图像数量
        """
        # 这个函数现在不需要做任何事情
        # 所有逻辑已经在 _save_image_to_gallery 中处理
        pass
    
    def _prepare_ui_images(self) -> List[Dict]:
        """
        准备UI展示的图像列表
        
        Returns:
            UI图像信息列表
        """
        ui_images = []
        
        for metadata in self.image_metadata:
            # 检查文件是否存在
            if os.path.exists(metadata["temp_path"]):
                ui_images.append({
                    "filename": metadata["filename"],
                    "subfolder": "image_accumulator",  # 相对于temp目录的子文件夹
                    "type": "temp",
                    # 额外信息（可在前端使用）
                    "index": metadata["index"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                })
        
        return ui_images
    
    def _select_single_image(self, selected_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择单张图像输出
        
        Args:
            selected_index: 图像索引（-1表示最新）
            
        Returns:
            (图像张量 [1, H, W, C], 遮罩张量 [1, H, W])
        """
        # 检查图像库是否为空
        if len(self.image_metadata) == 0:
            print("[ImageAccumulator] ⚠️  图像库为空，返回空图像")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.ones((1, 64, 64), dtype=torch.float32)  # 全白遮罩
            return empty_image, empty_mask
        
        # 处理索引
        if selected_index == -1:
            selected_index = len(self.image_metadata) - 1  # 最新图像
        else:
            # 确保索引在有效范围内
            selected_index = max(0, min(selected_index, len(self.image_metadata) - 1))
        
        # 获取元数据
        metadata = self.image_metadata[selected_index]
        
        # 从文件读取图像
        try:
            image = self._load_image_from_file(metadata["temp_path"])
            
            # 单张图像没有填充，创建全白遮罩（表示全部是原始图像）
            h, w = image.shape[1], image.shape[2]
            mask = torch.ones((1, h, w), dtype=torch.float32)
            
            print(f"[ImageAccumulator] 📤 输出单张图像 #{metadata['index']}: {metadata['width']}x{metadata['height']}")
            return image, mask
            
        except Exception as e:
            print(f"[ImageAccumulator] ❌ 加载图像失败: {e}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.ones((1, 64, 64), dtype=torch.float32)
            return empty_image, empty_mask
    
    def _select_batch_images(self, batch_start: int, batch_end: int, align_mode: str, pad_color: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择批次图像输出（智能对齐）
        
        Args:
            batch_start: 起始索引
            batch_end: 结束索引
            align_mode: 对齐模式（first/largest/smallest）
            pad_color: 填充颜色（十六进制）
            
        Returns:
            (批次图像张量 [N, H, W, C], 批次遮罩张量 [N, H, W])
        """
        # 检查图像库是否为空
        if len(self.image_metadata) == 0:
            print("[ImageAccumulator] ⚠️  图像库为空，返回空图像")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.ones((1, 64, 64), dtype=torch.float32)
            return empty_image, empty_mask
        
        # 确保索引在有效范围内
        batch_start = max(0, min(batch_start, len(self.image_metadata) - 1))
        batch_end = max(0, min(batch_end, len(self.image_metadata) - 1))
        
        # 确保 start <= end
        if batch_start > batch_end:
            batch_start, batch_end = batch_end, batch_start
        
        # 获取批次索引范围
        indices = list(range(batch_start, batch_end + 1))
        
        print(f"[ImageAccumulator] 📦 批次输出: 索引 {batch_start}-{batch_end} ({len(indices)}张)")
        
        # 加载所有图像
        images = []
        for idx in indices:
            if idx < len(self.image_metadata):
                metadata = self.image_metadata[idx]
                try:
                    img = self._load_image_from_file(metadata["temp_path"])
                    images.append(img)
                except Exception as e:
                    print(f"[ImageAccumulator] ⚠️  加载图像#{idx}失败: {e}")
        
        if not images:
            print("[ImageAccumulator] ❌ 没有成功加载任何图像")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.ones((1, 64, 64), dtype=torch.float32)
            return empty_image, empty_mask
        
        # 确定目标尺寸
        target_height, target_width = self._get_target_size(images, align_mode)
        
        print(f"[ImageAccumulator] 🎯 对齐模式: {align_mode}, 目标尺寸: {target_width}x{target_height}")
        
        # 对齐所有图像并生成遮罩
        aligned_images = []
        aligned_masks = []
        
        for i, img in enumerate(images):
            aligned_img, mask = self._align_image_with_mask(img, target_height, target_width, pad_color, generate_mask=True)
            aligned_images.append(aligned_img)
            aligned_masks.append(mask)
        
        # 合并为batch
        batch_images = torch.cat(aligned_images, dim=0)
        batch_masks = torch.cat(aligned_masks, dim=0)
        
        print(f"[ImageAccumulator] ✅ 批次输出完成: images={batch_images.shape}, masks={batch_masks.shape}")
        
        return batch_images, batch_masks
    
    def _get_target_size(self, images: List[torch.Tensor], align_mode: str) -> Tuple[int, int]:
        """
        获取目标对齐尺寸
        
        Args:
            images: 图像列表
            align_mode: 对齐模式
            
        Returns:
            (target_height, target_width)
        """
        if align_mode == "first":
            # 使用第一张图像的尺寸
            return images[0].shape[1], images[0].shape[2]
        
        elif align_mode == "largest":
            # 使用最大尺寸
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            return max_h, max_w
        
        elif align_mode == "smallest":
            # 使用最小尺寸（会裁剪）
            min_h = min(img.shape[1] for img in images)
            min_w = min(img.shape[2] for img in images)
            return min_h, min_w
        
        else:
            # 默认使用最大尺寸
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            return max_h, max_w
    
    def _align_image_with_mask(self, image: torch.Tensor, target_h: int, target_w: int, pad_color: str, generate_mask: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐图像到目标尺寸并生成遮罩（居中填充或裁剪）
        
        Args:
            image: 图像张量 [1, H, W, C]
            target_h: 目标高度
            target_w: 目标宽度
            pad_color: 填充颜色（十六进制，如 #000000）
            generate_mask: 是否生成遮罩
            
        Returns:
            (对齐后的图像 [1, H, W, C], 遮罩 [1, H, W])
        """
        current_h, current_w = image.shape[1], image.shape[2]
        
        # 如果尺寸已经匹配，直接返回
        if current_h == target_h and current_w == target_w:
            # 全白遮罩（全部是原始图像）
            mask = torch.ones((1, target_h, target_w), dtype=torch.float32)
            return image, mask
        
        # 计算填充或裁剪量
        pad_top = max(0, (target_h - current_h) // 2)
        pad_bottom = max(0, target_h - current_h - pad_top)
        pad_left = max(0, (target_w - current_w) // 2)
        pad_right = max(0, target_w - current_w - pad_left)
        
        # 解析十六进制颜色值
        color_value = self._parse_hex_color(pad_color)
        
        # 如果需要填充
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # 填充图像
            padded = torch.nn.functional.pad(
                image, 
                (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=color_value
            )
            
            # 生成遮罩：原始图像区域为白色(1.0)，填充区域为黑色(0.0)
            mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            mask[0, pad_top:pad_top+current_h, pad_left:pad_left+current_w] = 1.0
            
            return padded, mask
        
        # 如果需要裁剪
        else:
            crop_top = (current_h - target_h) // 2
            crop_left = (current_w - target_w) // 2
            cropped = image[:, crop_top:crop_top+target_h, crop_left:crop_left+target_w, :]
            
            # 裁剪后的遮罩全白（全部是原始图像）
            mask = torch.ones((1, target_h, target_w), dtype=torch.float32)
            
            return cropped, mask
    
    def _parse_hex_color(self, hex_color: str) -> float:
        """
        解析十六进制颜色为灰度值
        
        Args:
            hex_color: 十六进制颜色（如 #000000, #FFFFFF）
            
        Returns:
            灰度值 0.0-1.0
        """
        try:
            # 移除 # 号
            hex_color = hex_color.lstrip('#')
            
            # 处理简写形式 (如 #000 -> #000000)
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            
            # 解析 RGB
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            
            # 转换为灰度（使用标准公式）
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            
            return gray
            
        except Exception as e:
            print(f"[ImageAccumulator] ⚠️  解析颜色失败 '{hex_color}': {e}，使用黑色")
            return 0.0  # 默认黑色
    
    def _load_image_from_file(self, filepath: str) -> torch.Tensor:
        """
        从文件加载图像为张量
        
        Args:
            filepath: 图像文件路径
            
        Returns:
            图像张量 [1, H, W, C]
        """
        img = Image.open(filepath)
        img = img.convert("RGB")
        
        # 转换为numpy数组
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 转换为torch张量
        image = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]
        
        return image
    
    def _clear_gallery(self):
        """清空图像库（删除所有文件和元数据）"""
        print(f"[ImageAccumulator] 清空图像库，共 {len(self.image_metadata)} 张图像")
        
        # 删除所有临时文件
        for metadata in self.image_metadata:
            if os.path.exists(metadata["temp_path"]):
                try:
                    os.remove(metadata["temp_path"])
                except Exception as e:
                    print(f"[ImageAccumulator] 删除文件失败: {e}")
        
        # 清空元数据
        self.image_metadata.clear()
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """每次都执行（因为需要累积图像）"""
        return float("nan")  # 总是标记为已改变


# 注册节点
NODE_CLASS_MAPPINGS = {
    "YCImageAccumulator": YCImageAccumulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCImageAccumulator": "YC Image Accumulator",
}