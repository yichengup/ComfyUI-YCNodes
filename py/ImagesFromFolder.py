import os
import torch
import numpy as np
import random
from PIL import Image, ImageOps
import folder_paths
from nodes import LoadImage
import torchvision.transforms as transforms
import re
import comfy.utils
from PIL.PngImagePlugin import PngInfo
import json

# 定义允许的文件扩展名
ALLOWED_EXT = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif']

class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        
        return {"required": {
                    "path_type": (["input_dir", "custom_path"],),
                    "mode": (["single", "incremental", "random", "batch"],),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "optional": {
                    "image_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                    "folder_name": (["<root>"] + sorted(folders),),
                    "custom_path": ("STRING", {"default": "C:/Users/Pictures"}),
                    "label": ("STRING", {"default": "batch 001", "multiline": False}),
                    "output_filename": (["true", "false"],),
                    "allow_RGBA": (["false", "true"],),
                    "image_pattern": ("STRING", {"default": "*.png;*.jpg;*.jpeg;*.webp"}),
                }}
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_images"
    CATEGORY = "YCNode/Image"

    # 存储递增模式的状态 - 现在基于标签而不是目录ID
    batch_states = {}
    # 存储文件名缓存
    filename_cache = {}

    def load_images(self, path_type, mode, seed=0, image_index=0, folder_name="<root>", 
                   custom_path="", image_pattern="*.png;*.jpg;*.jpeg;*.webp",
                   label="batch 001", output_filename="true", allow_RGBA="false"):
        # 确定目标目录
        if path_type == "input_dir":
            input_dir = folder_paths.get_input_directory()
            if folder_name == "<root>":
                target_dir = input_dir
            else:
                target_dir = os.path.join(input_dir, folder_name)
        else:  # custom_path
            if not os.path.exists(custom_path):
                print(f"警告: 自定义路径 {custom_path} 不存在!")
                empty = torch.zeros((1, 64, 64, 3))
                return (empty, "")
            target_dir = custom_path

        # 生成批次信息
        batch_key = label

        # 处理文件名开关，如果设置为false，清空标签的文件名缓存
        if output_filename == "false":
            if batch_key in self.filename_cache:
                del self.filename_cache[batch_key]
            filename_output = ""

        # 获取所有匹配的图片文件
        patterns = image_pattern.split(";")
        image_files = []
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern.startswith("*."):
                ext = pattern[2:].lower()
                image_files.extend([f for f in os.listdir(target_dir) 
                                    if os.path.isfile(os.path.join(target_dir, f)) and
                                    f.lower().endswith(f".{ext}")])
        
        image_files = sorted(image_files)
        if not image_files:
            print(f"警告: 在 {target_dir} 中未找到匹配的图片")
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, "")

        # 如果是第一次使用该标签或批次信息变更，初始化状态
        if batch_key not in self.batch_states:
            self.batch_states[batch_key] = {
                "path": target_dir,
                "pattern": image_pattern,
                "index": 0
            }
        else:
            # 如果路径或模式变了，重置索引
            state = self.batch_states[batch_key]
            if state["path"] != target_dir or state["pattern"] != image_pattern:
                state["path"] = target_dir
                state["pattern"] = image_pattern
                state["index"] = 0
                print(f"标签 '{label}' 的路径或模式已更改，索引已重置。")

        def load_single_image(image_path):
            try:
                img = Image.open(image_path)
                
                # 处理RGBA图片
                if img.mode == 'RGBA' and allow_RGBA == "true":
                    # 保留RGBA格式
                    img_rgb = img
                    has_alpha = True
                elif img.mode == 'RGBA':
                    # 转换RGBA为RGB
                    img_rgb = img.convert('RGB')
                    has_alpha = False
                else:
                    img_rgb = img.convert('RGB')
                    has_alpha = False
                
                # 获取不带扩展名的文件名
                basename = os.path.basename(image_path)
                filename = os.path.splitext(basename)[0]
                
                # 转换为tensor，保持原始尺寸
                if has_alpha:
                    # 处理RGBA图像，转换为具有4个通道的tensor
                    img_array = np.array(img_rgb).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(img_array)[None,]
                else:
                    # 处理RGB图像
                    img_array = np.array(img_rgb).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(img_array)[None,]
                
                return image_tensor, filename
            except Exception as e:
                print(f"加载图片时出错 {image_path}: {str(e)}")
                return None, ""

        if mode == "batch":
            # 批处理模式 - 加载所有图片
            images = []
            filenames = []
            for img_file in image_files:
                image_path = os.path.join(target_dir, img_file)
                img_tensor, filename = load_single_image(image_path)
                if img_tensor is not None:
                    images.append(img_tensor)
                    if output_filename == "true":
                        filenames.append(filename)
            
            if images:
                batched_images = torch.cat(images, dim=0)
                if output_filename == "true":
                    filename_output = ";".join(filenames)
                    # 缓存文件名
                    self.filename_cache[batch_key] = filename_output
                else:
                    filename_output = ""
                return (batched_images, filename_output)
            else:
                empty = torch.zeros((1, 64, 64, 3))
                return (empty, "")
        else:
            # 单图模式 (single, incremental, random)
            if mode == "single":
                # 使用指定索引
                sel_index = min(max(0, image_index), len(image_files) - 1)
            elif mode == "random":
                # 使用随机索引
                random.seed(seed)
                sel_index = random.randint(0, len(image_files) - 1)
                print(f"标签 '{label}' 随机选择图片索引: {sel_index} (种子: {seed})")
            else:  # incremental
                # 使用基于标签的递增索引
                sel_index = self.batch_states[batch_key]["index"]
                # 更新下一次的索引
                next_index = (sel_index + 1) % len(image_files)
                self.batch_states[batch_key]["index"] = next_index
                print(f"标签 '{label}' 递增图片索引: {sel_index} (下一次: {next_index})")
            
            image_path = os.path.join(target_dir, image_files[sel_index])
            img_tensor, filename = load_single_image(image_path)
            if img_tensor is None:
                empty = torch.zeros((1, 64, 64, 3))
                return (empty, "")
            
            if output_filename == "true":
                # 更新文件名缓存
                self.filename_cache[batch_key] = filename
                return (img_tensor, filename)
            else:
                return (img_tensor, "")

class YC_Image_Save:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "output")
        self.type = 'output'
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": '', "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default":"_"}),
                "filename_number_padding": ("INT", {"default":4, "min":1, "max":9, "step":1}),
                "filename_number_start": (["false", "true"],),
                "extension": (['png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'bmp'], ),
            },
            "optional": {
                "caption": ("STRING", {"forceInput": True}),
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "文本文件的扩展名"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "filenames",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False)

    FUNCTION = "yc_save_images"

    OUTPUT_NODE = True

    CATEGORY = "YCNode/Image"

    def yc_save_images(self, images, output_path='', filename_prefix="ComfyUI", filename_delimiter='_',
                        extension='png', filename_number_padding=4, filename_number_start='false', 
                        caption=None, caption_file_extension=".txt",
                        prompt=None, extra_pnginfo=None):

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        
        # 使用固定值替代之前的参数
        quality = 100  # 固定使用最高质量
        overwrite_mode = 'false'  # 默认不覆盖

        # 处理输出路径
        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)

        # 确保输出目录存在
        if not os.path.exists(output_path):
            print(f"警告: 路径 {output_path} 不存在，正在创建目录。")
            os.makedirs(output_path, exist_ok=True)

        # 查找现有的计数器值
        try:
            if filename_number_start == 'true':
                pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
            else:
                pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"
            
            existing_counters = []
            for filename in os.listdir(output_path):
                match = re.match(pattern, filename)
                if match:
                    try:
                        existing_counters.append(int(match.group(1)))
                    except (ValueError, IndexError):
                        pass
                        
            existing_counters.sort(reverse=True)
            counter = existing_counters[0] + 1 if existing_counters else 1
        except Exception as e:
            print(f"警告: 计数器初始化失败: {str(e)}，使用默认值1")
            counter = 1

        # 设置扩展名
        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            print(f"警告: 扩展名 {extension} 无效。有效格式为: {', '.join([ext[1:] for ext in ALLOWED_EXT])}")
            file_extension = ".png"

        results = []
        output_filenames = []  # 存储文件名（而非完整路径）
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 处理元数据
            if extension == 'webp':
                img_exif = img.getexif()
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                exif_data = metadata

            # 生成文件名 (固定使用overwrite_mode='false')
            if filename_number_start == 'true':
                file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                base_filename = f"{counter:0{number_padding}}{delimiter}{filename_prefix}"
            else:
                file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                base_filename = f"{filename_prefix}{delimiter}{counter:0{number_padding}}"
            
            if os.path.exists(os.path.join(output_path, file)):
                counter += 1
                # 重新生成文件名
                if filename_number_start == 'true':
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                    base_filename = f"{counter:0{number_padding}}{delimiter}{filename_prefix}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                    base_filename = f"{filename_prefix}{delimiter}{counter:0{number_padding}}"

            # 保存图片
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file, quality=quality)
                elif extension == 'webp':
                    img.save(output_file, quality=quality, exif=exif_data)
                elif extension == 'png':
                    img.save(output_file, pnginfo=exif_data)
                elif extension == 'bmp':
                    img.save(output_file)
                elif extension == 'tiff':
                    img.save(output_file, quality=quality)
                else:
                    img.save(output_file, pnginfo=exif_data)

                print(f"图片保存至: {output_file}")
                output_filenames.append(file)  # 只添加文件名，不含路径
                
                # 保存文本描述（如果提供）
                if caption is not None and caption.strip() != "":
                    txt_file = base_filename + caption_file_extension
                    txt_path = os.path.abspath(os.path.join(output_path, txt_file))
                    try:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                        print(f"文本描述保存至: {txt_path}")
                    except Exception as e:
                        print(f"保存文本描述失败: {str(e)}")
                
                # 添加到结果
                results.append({
                    "filename": file,
                    "subfolder": "",
                    "type": self.type
                })

            except OSError as e:
                print(f'保存文件失败: {output_file}, 错误: {str(e)}')
            except Exception as e:
                print(f'保存文件失败，错误: {str(e)}')

            counter += 1

        # 返回结果 - 修改为返回文件名而不是完整路径
        return {"ui": {"images": results}, "result": (images, output_filenames,)}

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadImagesFromFolder": LoadImagesFromFolder,
    "YC_Image_Save": YC_Image_Save
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromFolder": "Load Images From Folder YC",
    "YC_Image_Save": "Image Save YC"
}