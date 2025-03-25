import torch
import numpy as np
import cv2

class MaskCrop_YC:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_frame": ("MASK",),
                "top_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "bottom_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "left_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "right_padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "round_to_multiple": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "detection_method": (["mask_area"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "COORDS")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_frame", "crop_frame_preview", "crop_coords")
    FUNCTION = "crop_with_mask"
    CATEGORY = "YCNode/Mask"

    def crop_with_mask(self, image, crop_frame, top_padding, bottom_padding, left_padding, right_padding, round_to_multiple, invert_mask, detection_method, mask=None):
        # 如果没有提供mask，创建一个全零mask
        if mask is None:
            # 创建一个与图像尺寸相同的全零mask
            mask = torch.zeros((1, image.shape[2], image.shape[3]), device=image.device)
        
        # 确保输入是正确的格式
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(crop_frame, torch.Tensor):
            crop_frame = crop_frame.cpu().numpy()
            
        # 处理mask维度
        if len(mask.shape) == 3:
            mask = mask[0]
        elif len(mask.shape) == 4:
            mask = mask[0, 0]
            
        if len(crop_frame.shape) == 3:
            crop_frame = crop_frame[0]
        elif len(crop_frame.shape) == 4:
            crop_frame = crop_frame[0, 0]

        # 如果需要反转遮罩
        if invert_mask:
            crop_frame = 1 - crop_frame

        # 转换为二值图像
        binary_mask = (crop_frame > 0.5).astype(np.uint8)
        
        # 找到mask的边界框
        if detection_method == "mask_area":
            y_indices, x_indices = np.nonzero(binary_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                # 创建一个空的预览图像，直接使用原始图像
                original_img_np = image[0].cpu().numpy()  # [H, W, C]
                if original_img_np.dtype != np.float32:
                    original_img_np = original_img_np.astype(np.float32)
                if original_img_np.max() > 1.0:
                    original_img_np = original_img_np / 255.0
                    
                # 直接使用原始图像，不添加任何标注
                empty_preview_tensor = torch.from_numpy(original_img_np).float()
                empty_preview_tensor = empty_preview_tensor.unsqueeze(0)  # 添加批次维度
                return (image, mask, crop_frame, empty_preview_tensor, (0, 0, image.shape[3], image.shape[2]))
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 添加padding
            x_min = max(0, x_min - left_padding)
            x_max = min(binary_mask.shape[1], x_max + right_padding)
            y_min = max(0, y_min - top_padding)
            y_max = min(binary_mask.shape[0], y_max + bottom_padding)
            
            # 确保尺寸是round_to_multiple的倍数
            width = x_max - x_min
            height = y_max - y_min
            
            new_width = ((width + round_to_multiple - 1) // round_to_multiple) * round_to_multiple
            new_height = ((height + round_to_multiple - 1) // round_to_multiple) * round_to_multiple
            
            # 调整padding以达到所需尺寸
            x_pad = new_width - width
            y_pad = new_height - height
            
            x_min = max(0, x_min - x_pad // 2)
            x_max = min(binary_mask.shape[1], x_max + (x_pad - x_pad // 2))
            y_min = max(0, y_min - y_pad // 2)
            y_max = min(binary_mask.shape[0], y_max + (y_pad - y_pad // 2))
            
            # 裁剪图像和mask
            cropped_image = image[:, y_min:y_max, x_min:x_max, :]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            cropped_crop_frame = crop_frame[y_min:y_max, x_min:x_max]
            
            # 创建预览用的crop_frame - 改为图片格式，并绘制红色边框
            # 首先将图像转换到numpy进行处理
            original_img_np = image[0].cpu().numpy()  # [H, W, C]
            
            # 确保是RGB格式，且值范围为[0,1]
            if original_img_np.dtype != np.float32:
                original_img_np = original_img_np.astype(np.float32)
            if original_img_np.max() > 1.0:
                original_img_np = original_img_np / 255.0
                
            # 创建预览图像副本，不显示半透明遮罩，只显示边框
            preview_img = original_img_np.copy()
            
            # 找到原始mask的边界 - 未添加padding的原始区域
            orig_y_indices, orig_x_indices = np.nonzero(binary_mask)
            orig_x_min, orig_x_max = np.min(orig_x_indices), np.max(orig_x_indices)
            orig_y_min, orig_y_max = np.min(orig_y_indices), np.max(orig_y_indices)
            
            # 确保坐标是整数
            x_min_int, y_min_int = int(x_min), int(y_min)
            x_max_int, y_max_int = int(x_max), int(y_max)
            orig_x_min_int, orig_y_min_int = int(orig_x_min), int(orig_y_min)
            orig_x_max_int, orig_y_max_int = int(orig_x_max), int(orig_y_max)
            
            # 定义颜色
            red_color = np.array([1.0, 0.0, 0.0])    # 红色用于标注最终裁剪区域（含padding）
            green_color = np.array([0.0, 1.0, 0.0])  # 绿色用于标注多出的padding区域
            
            # 线宽
            line_width = 2
            
            # 绘制红色边框（最终裁剪区域，包含padding）
            # 顶部边框
            preview_img[y_min_int:y_min_int+line_width, x_min_int:x_max_int] = red_color
            # 底部边框
            preview_img[y_max_int-line_width:y_max_int, x_min_int:x_max_int] = red_color
            # 左侧边框
            preview_img[y_min_int:y_max_int, x_min_int:x_min_int+line_width] = red_color
            # 右侧边框
            preview_img[y_min_int:y_max_int, x_max_int-line_width:x_max_int] = red_color
            
            # 绘制绿色边框（原始遮罩区域，不含padding）
            # 只有当原始区域与裁剪区域不同时才绘制绿色边框
            if (orig_x_min_int != x_min_int or orig_y_min_int != y_min_int or 
                orig_x_max_int != x_max_int or orig_y_max_int != y_max_int):
                # 顶部边框
                preview_img[orig_y_min_int:orig_y_min_int+line_width, orig_x_min_int:orig_x_max_int] = green_color
                # 底部边框
                preview_img[orig_y_max_int-line_width:orig_y_max_int, orig_x_min_int:orig_x_max_int] = green_color
                # 左侧边框
                preview_img[orig_y_min_int:orig_y_max_int, orig_x_min_int:orig_x_min_int+line_width] = green_color
                # 右侧边框
                preview_img[orig_y_min_int:orig_y_max_int, orig_x_max_int-line_width:orig_x_max_int] = green_color
            
            # 转换回PyTorch张量格式 [B, H, W, C]
            preview_img_tensor = torch.from_numpy(preview_img).float()
            preview_img_tensor = preview_img_tensor.unsqueeze(0)  # 添加批次维度
            
            # 转换回torch tensor
            cropped_mask = torch.from_numpy(cropped_mask).float()
            cropped_crop_frame = torch.from_numpy(cropped_crop_frame).float()
            
            if len(cropped_mask.shape) == 2:
                cropped_mask = cropped_mask.unsqueeze(0)
            if len(cropped_crop_frame.shape) == 2:
                cropped_crop_frame = cropped_crop_frame.unsqueeze(0)
                
            # 返回裁剪坐标 (x_min, y_min, x_max, y_max)
            crop_coords = (int(x_min), int(y_min), int(x_max), int(y_max))
            return (cropped_image, cropped_mask, cropped_crop_frame, preview_img_tensor, crop_coords)
        
        # 如果没有检测到有效区域，返回原始图像和空的预览图
        original_img_np = image[0].cpu().numpy()
        if original_img_np.dtype != np.float32:
            original_img_np = original_img_np.astype(np.float32)
        if original_img_np.max() > 1.0:
            original_img_np = original_img_np / 255.0
        
        # 直接使用原始图像，不添加任何标注
        empty_preview_tensor = torch.from_numpy(original_img_np).float()
        empty_preview_tensor = empty_preview_tensor.unsqueeze(0)  # 添加批次维度
        
        return (image, mask, crop_frame, empty_preview_tensor, (0, 0, image.shape[3], image.shape[2]))

class MaskCropRestore_YC:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "crop_coords": ("COORDS",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cropped_mask": ("MASK",),
                "crop_frame": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "inpaint_mask")
    FUNCTION = "restore_crop"
    CATEGORY = "YCNode/Mask"

    def restore_crop(self, background_image, cropped_image, crop_coords, invert_mask, cropped_mask=None, crop_frame=None):
        x_min, y_min, x_max, y_max = crop_coords
        
        # 正确解析图像维度
        # 图像形状: [batch, height, width, channels]
        batch_size = background_image.shape[0]
        bg_height = background_image.shape[1]
        bg_width = background_image.shape[2]
        channels = background_image.shape[3]
        
        # 检查图像维度是否合理
        if cropped_image.shape[1] < 10 or cropped_image.shape[2] < 10:
            # 修复坐标 - 确保至少有1像素的差距
            if x_min >= x_max:
                x_max = x_min + 1
            if y_min >= y_max:
                y_max = y_min + 1
        
        # 强制确保裁剪区域有合理的高度和宽度（至少10像素）
        if y_max - y_min < 10:
            height_to_add = 10 - (y_max - y_min)
            y_min = max(0, y_min - height_to_add // 2)
            y_max = min(bg_height, y_max + (height_to_add - height_to_add // 2))
            
        if x_max - x_min < 10:
            width_to_add = 10 - (x_max - x_min)
            x_min = max(0, x_min - width_to_add // 2)
            x_max = min(bg_width, x_max + (width_to_add - width_to_add // 2))
        
        # 确保坐标在有效范围内
        x_min = max(0, min(x_min, bg_width - 1))
        y_min = max(0, min(y_min, bg_height - 1))
        x_max = max(x_min + 1, min(x_max, bg_width))
        y_max = max(y_min + 1, min(y_max, bg_height))
        
        # 创建输出图像和mask（与背景图像相同大小）
        output_image = background_image.clone()
        output_mask = torch.zeros((1, bg_height, bg_width), device=background_image.device)
        
        # 确保裁剪图像尺寸与目标区域匹配
        expected_height = y_max - y_min
        expected_width = x_max - x_min
        
        # 验证裁剪图像尺寸
        if cropped_image.shape[1] != expected_height or cropped_image.shape[2] != expected_width:
            # 调整放置策略
            if expected_width == 0 or expected_height == 0:
                # 使用安全的默认值
                x_min, y_min = 0, 0
                x_max = min(cropped_image.shape[2], bg_width)
                y_max = min(cropped_image.shape[1], bg_height)
                expected_width = x_max - x_min
                expected_height = y_max - y_min
            else:
                # 尝试调整坐标以匹配实际图像尺寸
                src_height, src_width = cropped_image.shape[1], cropped_image.shape[2]
                
                # 如果输入图像尺寸小于目标区域，直接使用输入图像尺寸
                if src_height < expected_height:
                    y_max = y_min + src_height
                if src_width < expected_width:
                    x_max = x_min + src_width
                
                # 检查调整后尺寸是否合理
                if y_max - y_min < 10:
                    y_max = min(y_min + 10, bg_height)
                    
                if x_max - x_min < 10:
                    x_max = min(x_min + 10, bg_width)
                
                # 如果输入图像尺寸大于目标区域，可能需要裁剪输入图像
                expected_height = y_max - y_min
                expected_width = x_max - x_min
        
        # 将裁剪的图像放回原位置
        try:
            # 确保区域有效
            if expected_width <= 0 or expected_height <= 0:
                raise ValueError(f"无效的区域大小: 宽度={expected_width}, 高度={expected_height}")
                
            # 安全复制，考虑到输入图像可能与目标区域大小不匹配
            copy_height = min(expected_height, cropped_image.shape[1])
            copy_width = min(expected_width, cropped_image.shape[2])
            
            if copy_width > 0 and copy_height > 0:
                # 只复制有效的部分 - 修正索引
                output_image[:, y_min:y_min+copy_height, x_min:x_min+copy_width, :] = cropped_image[:, :copy_height, :copy_width, :]
                
        except RuntimeError:
            # 如果还是失败，尝试最保守的方法
            try:
                min_height = min(cropped_image.shape[1], bg_height - y_min)
                min_width = min(cropped_image.shape[2], bg_width - x_min)
                
                if min_height > 0 and min_width > 0:
                    output_image[:, y_min:y_min+min_height, x_min:x_min+min_width, :] = cropped_image[:, :min_height, :min_width, :]
                    # 更新复制区域的大小，供后面遮罩使用
                    copy_height, copy_width = min_height, min_width
            except:
                pass
        
        # 安全处理mask
        try:
            # 确定要使用的遮罩
            mask_to_use = None
            if cropped_mask is not None:
                mask_to_use = cropped_mask
            elif crop_frame is not None:
                mask_to_use = crop_frame
            else:
                # 如果两个遮罩都没有提供，创建一个全1遮罩（即选中整个裁剪区域）
                # 使用与裁剪图像一致的尺寸
                height, width = cropped_image.shape[1], cropped_image.shape[2]
                mask_to_use = torch.ones((1, height, width), device=cropped_image.device)
            
            # 将裁剪的遮罩转换为正确的格式
            if isinstance(mask_to_use, torch.Tensor):
                if len(mask_to_use.shape) == 4:
                    mask_to_use = mask_to_use[0]
                
                # 确保mask的维度至少是3维 [C, H, W]
                if len(mask_to_use.shape) == 2:
                    mask_to_use = mask_to_use.unsqueeze(0)
                
                # 验证遮罩维度
                if len(mask_to_use.shape) != 3:
                    return (output_image, output_mask)
                    
                # 验证遮罩尺寸
                mask_height, mask_width = mask_to_use.shape[1], mask_to_use.shape[2]
                
                # 检查遮罩尺寸是否异常小
                if mask_height < 10 or mask_width < 10:
                    # 如果遮罩高度或宽度异常小，尝试调整
                    if expected_height > 10 and expected_width > 10:
                        try:
                            # 创建新的遮罩并调整大小
                            new_mask = torch.nn.functional.interpolate(
                                mask_to_use.unsqueeze(0) if len(mask_to_use.shape) == 3 else mask_to_use,
                                size=(expected_height, expected_width),
                                mode='nearest'
                            )
                            mask_to_use = new_mask.squeeze(0) if len(mask_to_use.shape) == 3 else new_mask
                            mask_height, mask_width = expected_height, expected_width
                        except:
                            pass
                
                # 使用已经调整过的目标区域大小
                copy_height = min(copy_height if 'copy_height' in locals() else expected_height, mask_height)
                copy_width = min(copy_width if 'copy_width' in locals() else expected_width, mask_width)
                
                # 确保copy_height和copy_width至少有10像素（防止线状遮罩）
                copy_height = max(10, copy_height)
                copy_width = max(10, copy_width)
                
                # 确保不超出背景边界
                if y_min + copy_height > bg_height:
                    copy_height = bg_height - y_min
                if x_min + copy_width > bg_width:
                    copy_width = bg_width - x_min
                
                if copy_width > 0 and copy_height > 0:
                    # 确保索引有效
                    if copy_height > mask_to_use.shape[1]:
                        copy_height = mask_to_use.shape[1]
                    if copy_width > mask_to_use.shape[2]:
                        copy_width = mask_to_use.shape[2]
                    
                    # 安全复制遮罩 - 修正索引
                    try:
                        output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = mask_to_use[:, :copy_height, :copy_width]
                    except:
                        try:
                            # 强制重新调整遮罩尺寸
                            adjusted_mask = torch.nn.functional.interpolate(
                                mask_to_use.unsqueeze(0) if len(mask_to_use.shape) == 3 else mask_to_use,
                                size=(copy_height, copy_width),
                                mode='nearest'
                            )
                            adjusted_mask = adjusted_mask.squeeze(0) if len(mask_to_use.shape) == 3 else adjusted_mask
                            
                            # 再次尝试放置
                            output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = adjusted_mask
                        except:
                            pass
                    
                    # 验证输出遮罩
                    nonzero_count = torch.count_nonzero(output_mask)
                    if nonzero_count == 0:
                        # 最后尝试 - 创建一个简单的矩形遮罩
                        try:
                            output_mask[:, y_min:y_min+copy_height, x_min:x_min+copy_width] = 1.0
                        except:
                            pass
        except RuntimeError:
            # 如果还是失败，尝试最保守的方法
            try:
                if 'mask_to_use' in locals() and isinstance(mask_to_use, torch.Tensor):
                    mask_height, mask_width = mask_to_use.shape[1], mask_to_use.shape[2]
                    min_height = min(mask_height, bg_height - y_min)
                    min_width = min(mask_width, bg_width - x_min)
                    
                    # 确保最小高度和宽度不小于10像素
                    min_height = max(10, min_height)
                    min_width = max(10, min_width)
                    
                    # 确保不超出背景边界
                    if y_min + min_height > bg_height:
                        min_height = bg_height - y_min
                    if x_min + min_width > bg_width:
                        min_width = bg_width - x_min
                    
                    if min_height > 0 and min_width > 0:
                        # 确保不超出mask边界
                        mask_part = mask_to_use[:, :min(min_height, mask_height), :min(min_width, mask_width)]
                        # 确保目标区域足够大
                        if y_min + mask_part.shape[1] <= bg_height and x_min + mask_part.shape[2] <= bg_width:
                            output_mask[:, y_min:y_min+mask_part.shape[1], x_min:x_min+mask_part.shape[2]] = mask_part
            except:
                pass
                
            # 最后的尝试：创建一个简单的矩形遮罩
            try:
                valid_height = min(10, bg_height - y_min)
                valid_width = min(10, bg_width - x_min)
                if valid_height > 0 and valid_width > 0:
                    output_mask[:, y_min:y_min+valid_height, x_min:x_min+valid_width] = 1.0
            except:
                pass
        
        if invert_mask:
            output_mask = 1 - output_mask
        
        return (output_image, output_mask)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MaskCrop_YC": MaskCrop_YC,
    "MaskCropRestore_YC": MaskCropRestore_YC
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCrop_YC": "MaskCrop_YC",
    "MaskCropRestore_YC": "MaskCropRestore_YC"
} 