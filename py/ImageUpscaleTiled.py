import torch
import numpy as np
from PIL import Image
import os
import folder_paths
from spandrel import ModelLoader
import comfy.utils
import model_management

class ImageUpscaleTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                "rows": ("INT", { "default": 2, "min": 1, "max": 8, "step": 1 }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 8, "step": 1 }),
                "overlap": ("FLOAT", { "default": 0.1, "min": 0, "max": 0.5, "step": 0.01 }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_tiled"
    CATEGORY = "YiCheng/Image"

    def upscale_tiled(self, image, model_name, rows, cols, overlap):
        device = model_management.get_torch_device()
        
        # 1. 加载放大模型
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        model = ModelLoader().load_from_state_dict(sd)
        
        if not hasattr(model, 'model'):
            raise Exception("Invalid upscale model")
            
        upscale_model = model.model.eval()
        
        # 2. 计算分块参数
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        h = tile_h * rows
        w = tile_w * cols
        
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)

        # 限制最大重叠为tile大小的一半
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0

        # 3. 内存管理
        memory_required = model_management.module_size(upscale_model)
        memory_required += (tile_h * tile_w * 3) * image.element_size() * model.scale * model.scale * 2
        model_management.free_memory(memory_required, device)
        
        # 4. 移动模型到设备
        upscale_model = upscale_model.to(device)
        
        try:
            # 创建进度条
            total_tiles = rows * cols
            pbar = comfy.utils.ProgressBar(total_tiles)
            
            # 5. 分块处理
            tiles = []
            for i in range(rows):
                for j in range(cols):
                    y1 = i * tile_h
                    x1 = j * tile_w

                    if i > 0:
                        y1 -= overlap_h
                    if j > 0:
                        x1 -= overlap_w

                    y2 = y1 + tile_h + overlap_h
                    x2 = x1 + tile_w + overlap_w

                    if y2 > h:
                        y2 = h
                        y1 = y2 - tile_h - overlap_h
                    if x2 > w:
                        x2 = w
                        x1 = x2 - tile_w - overlap_w

                    # 提取tile
                    tile = image[:, y1:y2, x1:x2, :]
                    
                    # 转换格式并放大
                    tile = tile.movedim(-1,-3).to(device)
                    with torch.no_grad():
                        upscaled_tile = upscale_model(tile)
                    tiles.append(upscaled_tile)
                    
                    # 更新进度条
                    pbar.update(1)

            # 6. 合并tiles
            tiles = torch.cat(tiles, dim=0)
            
            # 7. 计算输出尺寸
            out_h = h * model.scale
            out_w = w * model.scale
            overlap_h_up = overlap_h * model.scale
            overlap_w_up = overlap_w * model.scale
            tile_h_up = tile_h * model.scale
            tile_w_up = tile_w * model.scale

            # 8. 创建输出tensor
            out = torch.zeros((1, tiles.shape[1], out_h, out_w), device=device, dtype=tiles.dtype)

            # 9. 合并tiles
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    y1 = i * tile_h_up
                    x1 = j * tile_w_up

                    if i > 0:
                        y1 -= overlap_h_up
                    if j > 0:
                        x1 -= overlap_w_up

                    y2 = y1 + tile_h_up + overlap_h_up
                    x2 = x1 + tile_w_up + overlap_w_up

                    if y2 > out_h:
                        y2 = out_h
                        y1 = y2 - tile_h_up - overlap_h_up
                    if x2 > out_w:
                        x2 = out_w
                        x1 = x2 - tile_w_up - overlap_w_up

                    # 创建渐变mask
                    mask = torch.ones((1, 1, tile_h_up+overlap_h_up, tile_w_up+overlap_w_up), device=device, dtype=tiles.dtype)
                    if i > 0 and overlap_h_up > 0:
                        mask[:, :, :overlap_h_up, :] *= torch.linspace(0, 1, overlap_h_up, device=device, dtype=tiles.dtype).view(1, 1, -1, 1)
                    if j > 0 and overlap_w_up > 0:
                        mask[:, :, :, :overlap_w_up] *= torch.linspace(0, 1, overlap_w_up, device=device, dtype=tiles.dtype).view(1, 1, 1, -1)

                    # 应用mask
                    tile = tiles[idx:idx+1]  # 保持 NCHW 格式
                    mask = mask.repeat(1, tile.shape[1], 1, 1)
                    tile = tile * mask
                    out[:, :, y1:y2, x1:x2] = out[:, :, y1:y2, x1:x2] * (1 - mask) + tile
                    idx += 1

            # 最后转换回原始格式
            out = out.movedim(1, -1)  # NCHW -> NHWC

        finally:
            # 10. 清理资源
            upscale_model.to("cpu")
            if str(device) == 'cuda':
                torch.cuda.empty_cache()

        return (out,)

NODE_CLASS_MAPPINGS = {
    "ImageUpscaleTiled": ImageUpscaleTiled
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageUpscaleTiled": "Image Upscale Tiled"
}