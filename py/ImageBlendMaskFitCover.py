import torch
import numpy as np
from PIL import Image, ImageChops

# 复用原节点的模式列表
BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "soft_light",
    "hard_light",
    "darken",
    "lighten",
    "color_dodge",
    "color_burn",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
]

VISIBILITY_BLEND_MODES = ["replace", "multiply", "add"]


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def get_mask_bbox(mask_img: Image.Image, threshold: int = 128):
    arr = np.array(mask_img)
    if arr.max() <= 1:
        arr = arr * 255
    arr = (arr >= threshold).astype(np.uint8)
    coords = np.argwhere(arr)
    if coords.shape[0] == 0:
        return (0, 0, mask_img.width, mask_img.height)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (x0, y0, x1 + 1, y1 + 1)


def resize_fg_cover_bbox(fg_img: Image.Image, bbox_size, rotation=0):
    """等比例放大至覆盖 bbox（无留白），然后旋转。"""
    fg_w, fg_h = fg_img.size
    box_w, box_h = bbox_size
    scale = max(box_w / max(1, fg_w), box_h / max(1, fg_h))
    new_w = max(1, int(fg_w * scale))
    new_h = max(1, int(fg_h * scale))
    fg_resized = fg_img.resize((new_w, new_h), Image.LANCZOS)
    if rotation != 0:
        fg_resized = fg_resized.convert("RGBA")
        fg_resized = fg_resized.rotate(rotation, Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    return fg_resized


def resize_vismask_match_bbox(mask_img: Image.Image, bbox_size, rotation=0):
    """将可见遮罩按 bbox 尺寸缩放并旋转，保持与前景同尺寸/同变换。"""
    mask_resized = mask_img.resize(bbox_size, Image.LANCZOS)
    if rotation != 0:
        mask_resized = mask_resized.rotate(rotation, Image.BICUBIC, expand=True, fillcolor=0)
    return mask_resized


class ImageBlendMaskFitCover:
    """
    定位遮罩1：确定位置并计算 bbox，前景等比例放大以覆盖 bbox（无留白）。
    """

    def __init__(self):
        self.NODE_NAME = "ImageBlendMaskFitCover"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "layer_mask": ("MASK",),
                "blend_mode": (BLEND_MODES,),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "反转定位遮罩1"}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1, "tooltip": "定位遮罩整体左右偏移（+为右，-为左，像素）"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1, "tooltip": "定位遮罩整体上下偏移（+为下，-为上，像素）"}),
                "bbox_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "定位 bbox 额外缩放系数（>1 放大，<1 缩小）"}),
            },
            "optional": {
                "layer_alpha_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_maskfit"
    CATEGORY = "YCNode/Image"

    def blend_maskfit(
        self,
        background_image,
        layer_image,
        layer_mask,
        blend_mode,
        rotation,
        opacity,
        invert_mask,
        offset_x,
        offset_y,
        bbox_scale,
        layer_alpha_mask=None,
    ):
        ret_images = []
        if layer_alpha_mask is not None:
            max_batch = max(
                len(background_image),
                len(layer_image),
                len(layer_mask),
                len(layer_alpha_mask),
            )
        else:
            max_batch = max(
                len(background_image),
                len(layer_image),
                len(layer_mask),
            )

        for i in range(max_batch):
            bg_tensor = background_image[i] if i < len(background_image) else background_image[-1]
            fg_tensor = layer_image[i] if i < len(layer_image) else layer_image[-1]
            mask_tensor = layer_mask[i] if i < len(layer_mask) else layer_mask[-1]
            if layer_alpha_mask is not None:
                alpha_mask_tensor = layer_alpha_mask[i] if i < len(layer_alpha_mask) else layer_alpha_mask[-1]
            else:
                alpha_mask_tensor = None

            bg_pil = tensor2pil(bg_tensor)
            fg_pil = tensor2pil(fg_tensor)
            mask_pil = tensor2pil(mask_tensor).convert("L")
            if alpha_mask_tensor is not None:
                alpha_mask_pil = tensor2pil(alpha_mask_tensor).convert("L")
            else:
                # 未提供 layer_alpha_mask 时，等价于全白遮罩（不改变原始前景透明度）
                alpha_mask_pil = Image.new("L", fg_pil.size, 255)

            # 将 layer_image 与新增遮罩先合成为“透明前景”
            if alpha_mask_pil.size != fg_pil.size:
                alpha_mask_pil = alpha_mask_pil.resize(fg_pil.size, Image.LANCZOS)
            fg_rgba = fg_pil.convert("RGBA")
            r, g, b, a = fg_rgba.split()
            combined_fg_alpha = ImageChops.multiply(a, alpha_mask_pil)
            fg_rgba = Image.merge("RGBA", (r, g, b, combined_fg_alpha))

            # 尺寸适配到背景
            if mask_pil.size != bg_pil.size:
                mask_pil = mask_pil.resize(bg_pil.size, Image.NEAREST)

            if invert_mask:
                mask_pil = Image.fromarray(255 - np.array(mask_pil))

            bbox = get_mask_bbox(mask_pil)
            box_x0, box_y0, box_x1, box_y1 = bbox
            box_w = box_x1 - box_x0
            box_h = box_y1 - box_y0

            # 在定位遮罩的 bbox 基础上，应用额外缩放和偏移，获得更可控的放置区域
            if box_w > 0 and box_h > 0:
                bg_w, bg_h = bg_pil.size
                # 原 bbox 中心
                cx = box_x0 + box_w * 0.5
                cy = box_y0 + box_h * 0.5

                # 应用缩放
                scaled_w = max(1, int(box_w * bbox_scale))
                scaled_h = max(1, int(box_h * bbox_scale))

                # 应用偏移（像素）
                cx = cx + offset_x
                cy = cy + offset_y

                # 重新计算缩放+偏移后的 bbox，并限制在背景范围内
                new_x0 = int(round(cx - scaled_w * 0.5))
                new_y0 = int(round(cy - scaled_h * 0.5))
                new_x1 = new_x0 + scaled_w
                new_y1 = new_y0 + scaled_h

                new_x0 = max(0, min(new_x0, bg_w - 1))
                new_y0 = max(0, min(new_y0, bg_h - 1))
                new_x1 = max(new_x0 + 1, min(new_x1, bg_w))
                new_y1 = max(new_y0 + 1, min(new_y1, bg_h))

                box_x0, box_y0, box_x1, box_y1 = new_x0, new_y0, new_x1, new_y1
                box_w = box_x1 - box_x0
                box_h = box_y1 - box_y0

            # 前景等比例放大覆盖定位 bbox
            fg_resized = resize_fg_cover_bbox(fg_rgba, (box_w, box_h), rotation)

            # 计算中心对齐的粘贴位置
            paste_x = box_x0 + (box_w - fg_resized.width) // 2
            paste_y = box_y0 + (box_h - fg_resized.height) // 2

            fg_canvas = Image.new("RGBA", bg_pil.size, (0, 0, 0, 0))
            fg_resized_rgba = fg_resized.convert("RGBA")

            fg_canvas.paste(fg_resized_rgba, (paste_x, paste_y), fg_resized_rgba)

            # 使用定位遮罩限制显示：将遮罩缩放到背景尺寸后与当前前景 alpha 相乘
            if mask_pil.size != bg_pil.size:
                mask_bg = mask_pil.resize(bg_pil.size, Image.LANCZOS)
            else:
                mask_bg = mask_pil
            current_alpha = fg_canvas.split()[-1]
            combined_alpha = ImageChops.multiply(current_alpha, mask_bg)
            fg_canvas.putalpha(combined_alpha)

            # alpha 合成，透明区露出背景
            out = Image.alpha_composite(bg_pil.convert("RGBA"), fg_canvas)
            ret_images.append(pil2tensor(out.convert("RGB")))

        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {"ImageBlendMaskFitCover": ImageBlendMaskFitCover}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBlendMaskFitCover": "Image Blend MaskFit Cover"}

