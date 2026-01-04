import torch
import torchvision.transforms as T
import comfy.model_management

class YCMaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 6, "min": 0, "max": 256, "step": 1, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "YCNode/Mask"

    def execute(self, mask, amount, device):
        if amount == 0:
            return (mask,)

        if "gpu" == device:
            mask = mask.to(comfy.model_management.get_torch_device())
        elif "cpu" == device:
            mask = mask.to('cpu')

        if amount % 2 == 0:
            amount+= 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

        if "gpu" == device or "cpu" == device:
            mask = mask.to(comfy.model_management.intermediate_device())

        return(mask,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YCMaskBlur": YCMaskBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCMaskBlur": "Mask Blur (YC)"
}

