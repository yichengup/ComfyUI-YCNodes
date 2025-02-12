import torch
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageSequence
import os
from pathlib import Path
import folder_paths
import hashlib

class ImageLoaderAdvanced:
    @classmethod
    def get_all_files(cls):
        """获取所有可用文件的列表"""
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return sorted(files)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (s.get_all_files(), {
                    "image_upload": True,
                }),
                "mask_mode": (["none", "alpha"], {"default": "alpha"}),
                "mask_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "mask_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "YCNode/Image"
    
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False)
    OUTPUT_NODE = True

    def load_image(self, image, mask_mode="alpha", mask_blur=0.0, mask_strength=1.0):
        try:
            # 获取图片路径
            image_path = folder_paths.get_annotated_filepath(image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 使用PIL加载图像
            img = Image.open(image_path)
            
            output_images = []
            output_masks = []
            w, h = None, None
            
            # 处理图像序列
            for i in ImageSequence.Iterator(img):
                i = ImageOps.exif_transpose(i)
                
                # 处理特殊模式
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                
                # 转换为RGB
                image = i.convert("RGB")
                
                # 记录第一帧的尺寸
                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]
                
                # 确保所有帧尺寸一致
                if image.size[0] != w or image.size[1] != h:
                    continue
                
                # 转换图像
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                # 处理遮罩
                if mask_mode == "alpha" and 'A' in i.getbands():
                    # 获取alpha通道
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    # 应用遮罩强度
                    mask = mask * mask_strength
                    # 应用高斯模糊
                    if mask_blur > 0:
                        kernel_size = int(mask_blur * 2) * 2 + 1
                        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), mask_blur)
                else:
                    mask = np.ones((h, w), dtype=np.float32)
                
                mask = torch.from_numpy(mask)
                
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))
            
            # 处理多帧图像
            if len(output_images) > 1 and img.format not in ['MPO']:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]
            
            return (output_image, output_mask)
            
        except Exception as e:
            print(f"Error loading image {image}: {str(e)}")
            raise e

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {
    "ImageLoaderAdvanced": ImageLoaderAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoaderAdvanced": "Load Image Advanced"
} 
