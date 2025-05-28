import torch
import os
import json
import numpy as np
from PIL import Image
import folder_paths
import re
import inspect

MAX_FLOW_NUM = 12  # 扩展到12个输入端口
lazy_options = {"lazy": True}  # 懒加载选项

# 创建通用类型处理容器
class AllContainer:
    def __contains__(self, item):
        return True

    def __getitem__(self, key):
        if key.startswith("input"):
            return "*", {"lazy": True, "forceInput": True}
        elif key == "cascade_input":
            return "*", {"lazy": True, "forceInput": True, "tooltip": "级联输入，通常连接到上一个选择器的输出"}
        elif key == "chain_input":
            return "*", {"lazy": True, "forceInput": True, "tooltip": "来自下一个选择器的链接输入"}
        else:
            return "*", {"lazy": True}

# 检查是否支持高级模型执行
def is_advanced_model_supported():
    try:
        stack = inspect.stack()
        if stack[2].function == 'get_input_info':
            return True
        return False
    except:
        return False

class textIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": MAX_FLOW_NUM-1, "step": 1}),
            },
            "optional": {
            }
        }
        # 动态添加12个文本输入端口
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["text%d" % i] = ("STRING", {**lazy_options, "forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "index_switch"

    CATEGORY = "YCNode/Logic"

    def check_lazy_status(self, index, **kwargs):
        key = "text%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "text%d" % index
        # 如果对应索引的输入端口未连接,返回空字符串
        return (kwargs.get(key, ""),)

class TextConditionSwitch:
    """
    文本条件判断器 - 根据输入文本是否与预设文本相匹配来选择输出A或B图像
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": False, "tooltip": "输入文本，将与预设文本比较"}),
                "preset_text": ("STRING", {"default": "", "multiline": False, "tooltip": "预设文本，用于与输入文本比较"}),
                "case_sensitive": ("BOOLEAN", {"default": True, "tooltip": "是否区分大小写"}),
            },
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "当输入文本与预设文本匹配时输出的图像"}),
                "image_b": ("IMAGE", {"tooltip": "当输入文本与预设文本不匹配时输出的图像"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN",)
    RETURN_NAMES = ("image", "is_matched",)
    FUNCTION = "condition_switch"
    CATEGORY = "YCNode/Logic"

    def condition_switch(self, input_text, preset_text, case_sensitive, image_a=None, image_b=None):
        # 判断文本是否匹配
        if case_sensitive:
            is_matched = input_text == preset_text
        else:
            is_matched = input_text.lower() == preset_text.lower()
            
        # 根据匹配结果选择输出图像
        if is_matched:
            output_image = image_a if image_a is not None else None
        else:
            output_image = image_b if image_b is not None else None
            
        # 如果没有图像可用，返回空图像
        if output_image is None:
            # 创建1x1的黑色图像
            empty_image = torch.zeros(1, 1, 1, 3)
            return (empty_image, is_matched)
            
        return (output_image, is_matched)

class extractNumberFromText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": False}),
                "default_value": ("INT", {"default": 0}),  # 当未找到数字时的默认值
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("number",)
    FUNCTION = "extract"
    CATEGORY = "YCNode/Logic"

    def extract(self, text, default_value):
        # 使用正则表达式匹配文本开头的数字
        match = re.match(r'^\d+', text.strip())
        if match:
            # 如果找到数字,转换为整数返回
            return (int(match.group()),)
        else:
            # 如果未找到数字,返回默认值
            return (default_value,)

class SuperIndexSelector:
    """
    LoRA选择器 - 专为LoRA切换设计的专用选择器
    
    特点：
    - 直观的"模式"选择，比索引值更容易理解
    - 只有10个LoRA输入端口，设计清晰
    - 专门的级联输入端口，支持串联多个选择器
    - 惰性加载，只加载被选择的LoRA，节省内存
    
    使用方法：
    1. 模式="选择LoRA"时，可以使用index选择input0-input9连接的LoRA
    2. 模式="使用级联"时，会使用cascade_input作为输出
    3. 将第一个选择器的输出连接到第二个选择器的cascade_input
    4. 第一个选择器设为"选择LoRA"，第二个设为"使用级联"
    
    可以无限级联多个选择器，管理大量LoRA模型
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "mode": (["选择LoRA", "使用级联"], {"default": "选择LoRA", "tooltip": "选择模式：选择LoRA使用index选择输入端口，使用级联则从级联输入获取LoRA"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1, "tooltip": "在'选择LoRA'模式下，选择0-9号LoRA"}),
            },
            "optional": {}
        }
        
        # 使用通用容器绕过类型验证
        if is_advanced_model_supported():
            inputs["optional"] = AllContainer()
        else:
            # 添加10个普通输入端口和级联输入端口
            inputs["optional"]["cascade_input"] = ("*", {"forceInput": True, "tooltip": "级联输入，通常连接到上一个选择器的输出"})
            for i in range(10):
                inputs["optional"]["input%d" % i] = ("*", {**lazy_options, "forceInput": True, "tooltip": f"LoRA输入 #{i+1}"})
            
        return inputs

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("selected_lora",)
    FUNCTION = "select_lora"
    CATEGORY = "YCNode/LoRA"

    def check_lazy_status(self, mode, index, **kwargs):
        if mode == "使用级联":
            if kwargs.get("cascade_input", None) is None:
                return ["cascade_input"]
        else:  # 选择LoRA模式
            key = "input%d" % index
            if kwargs.get(key, None) is None:
                return [key]

    def select_lora(self, mode, index, **kwargs):
        if mode == "使用级联":
            if "cascade_input" in kwargs and kwargs["cascade_input"] is not None:
                print(f"使用级联LoRA")
                return (kwargs["cascade_input"],)
            else:
                print(f"警告：级联模式已选择但级联输入未连接")
                return (None,)
        else:  # 选择LoRA模式
            key = "input%d" % index
            if key not in kwargs or kwargs[key] is None:
                print(f"警告：索引 {index} 未连接任何LoRA")
                return (None,)
            print(f"已选择LoRA #{index}")
            return (kwargs[key],)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # 总是更新

class DynamicThreshold:
    """
    Compares an input value to predefined ranges and outputs a corresponding value.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_value": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "compare"
    CATEGORY = "YCNode/Logic"

    def compare(self, input_value):
        if 1 <= input_value <= 150:
            return (7.0,)
        elif 151 <= input_value <= 200:
            return (5.3,)
        elif 201 <= input_value <= 250:
            return (4.5,)
        elif 251 <= input_value <= 300:
            return (3.8,)
        elif 301 <= input_value <= 350:
            return (3.3,)
        elif 351 <= input_value <= 400:
            return (2.9,)
        elif 401 <= input_value <= 450:
            return (2.8,)
        elif 451 <= input_value <= 500:
            return (2.6,)
        elif 501 <= input_value <= 550:
            return (2.2,) 
        elif 551 <= input_value <= 600:
            return (2.2,) 
        elif 601 <= input_value <= 650:
            return (1.8,)
        elif 651 <= input_value <= 700:
            return (1.7,)
        elif 701 <= input_value <= 750:
            return (1.6,)
        elif 751 <= input_value <= 800:
            return (1.5,)
        elif 801 <= input_value <= 900:
            return (1.4,)
        elif 901 <= input_value <= 950:
            return (1.3,)
        elif 951 <= input_value <= 1000:
            return (1.2,)
        elif 1001 <= input_value <= 1100:
            return (1.1,)
        elif 1101 <= input_value <= 1200:
            return (1,)
        elif 1201 <= input_value <= 1400:
            return (0.9,)
        elif 1401 <= input_value <= 1500:
            return (0.8,)
        elif 1501 <= input_value <= 1600:
            return (0.75,)
        else:
            return (0.6,)

class MaskConditionSwitch:
    """
    遮罩条件判断器 - 根据输入文本是否与预设文本相匹配来选择输出A或B遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": False, "tooltip": "输入文本，将与预设文本比较"}),
                "preset_text": ("STRING", {"default": "", "multiline": False, "tooltip": "预设文本，用于与输入文本比较"}),
                "case_sensitive": ("BOOLEAN", {"default": True, "tooltip": "是否区分大小写"}),
            },
            "optional": {
                "mask_a": ("MASK", {"tooltip": "当输入文本与预设文本匹配时输出的遮罩"}),
                "mask_b": ("MASK", {"tooltip": "当输入文本与预设文本不匹配时输出的遮罩"}),
            }
        }

    RETURN_TYPES = ("MASK", "BOOLEAN",)
    RETURN_NAMES = ("mask", "is_matched",)
    FUNCTION = "condition_switch"
    CATEGORY = "YCNode/Logic"

    def condition_switch(self, input_text, preset_text, case_sensitive, mask_a=None, mask_b=None):
        # 判断文本是否匹配
        if case_sensitive:
            is_matched = input_text == preset_text
        else:
            is_matched = input_text.lower() == preset_text.lower()
            
        # 根据匹配结果选择输出遮罩
        if is_matched:
            output_mask = mask_a if mask_a is not None else None
        else:
            output_mask = mask_b if mask_b is not None else None
            
        # 如果没有遮罩可用，返回空遮罩
        if output_mask is None:
            # 创建1x1的空遮罩
            empty_mask = torch.zeros(1, 1, 1, 1)
            return (empty_mask, is_matched)
            
        return (output_mask, is_matched)

class UniversalConditionGate:
    """
    通用条件开关 - 根据输入文本是否与预设文本相匹配来决定是否让输入通过
    适用于任何数据类型，可作为任意节点之间的条件门控
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 使用通用容器接受任何类型的输入
        if is_advanced_model_supported():
            return {
                "required": {
                    "input_text": ("STRING", {"default": "", "multiline": False, "tooltip": "输入文本，将与预设文本比较"}),
                    "preset_text": ("STRING", {"default": "", "multiline": False, "tooltip": "预设文本，用于与输入文本比较"}),
                    "case_sensitive": ("BOOLEAN", {"default": True, "tooltip": "是否区分大小写"}),
                    "invert": ("BOOLEAN", {"default": False, "tooltip": "反转逻辑：选择True时，不匹配则通过；选择False时，匹配则通过"}),
                },
                "optional": AllContainer()
            }
        else:
            return {
                "required": {
                    "input_text": ("STRING", {"default": "", "multiline": False, "tooltip": "输入文本，将与预设文本比较"}),
                    "preset_text": ("STRING", {"default": "", "multiline": False, "tooltip": "预设文本，用于与输入文本比较"}),
                    "case_sensitive": ("BOOLEAN", {"default": True, "tooltip": "是否区分大小写"}),
                    "invert": ("BOOLEAN", {"default": False, "tooltip": "反转逻辑：选择True时，不匹配则通过；选择False时，匹配则通过"}),
                },
                "optional": {
                    "input": ("*", {"tooltip": "任意类型的输入，当条件匹配时将被传递到输出"})
                }
            }

    RETURN_TYPES = ("*", "BOOLEAN",)
    RETURN_NAMES = ("output", "is_matched",)
    FUNCTION = "gate_control"
    CATEGORY = "YCNode/Logic"

    def check_lazy_status(self, input_text, preset_text, case_sensitive, invert, **kwargs):
        # 检查输入端口的连接状态
        if "input" not in kwargs or kwargs["input"] is None:
            return ["input"]
        return None

    def gate_control(self, input_text, preset_text, case_sensitive, invert, **kwargs):
        # 判断文本是否匹配
        if case_sensitive:
            is_matched = input_text == preset_text
        else:
            is_matched = input_text.lower() == preset_text.lower()
        
        # 如果设置了反转，则翻转匹配结果
        should_pass = is_matched if not invert else not is_matched
        
        # 检查是否有输入
        if "input" not in kwargs or kwargs["input"] is None:
            print("警告：通用条件开关 - 未连接输入")
            # 返回None和匹配状态
            return (None, is_matched)
        
        # 根据匹配结果决定是否传递输入
        if should_pass:
            return (kwargs["input"], is_matched)
        else:
            # 不匹配时返回None
            return (None, is_matched)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YC Text Index Switch": textIndexSwitch,
    "YC Extract Number": extractNumberFromText,
    "YC Super Selector": SuperIndexSelector,
    "DynamicThreshold": DynamicThreshold,
    "YC Text Condition Switch": TextConditionSwitch,
    "YC Mask Condition Switch": MaskConditionSwitch,
    "YC Universal Gate": UniversalConditionGate,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YC Text Index Switch": "Text Index Switch (YC)",
    "YC Extract Number": "Extract Number (YC)",
    "YC Super Selector": "LoRA Selector (YC)",
    "DynamicThreshold": "Dynamic Threshold",
    "YC Text Condition Switch": "Text Condition Switch (YC)",
    "YC Mask Condition Switch": "Mask Condition Switch (YC)",
    "YC Universal Gate": "Universal Condition Gate (YC)",
} 
