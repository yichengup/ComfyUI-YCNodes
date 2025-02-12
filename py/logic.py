import torch
import os
import json
import numpy as np
from PIL import Image
import folder_paths
import re

MAX_FLOW_NUM = 12  # 扩展到12个输入端口
lazy_options = {"lazy": True}  # 懒加载选项

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

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YC Text Index Switch": textIndexSwitch,
    "YC Extract Number": extractNumberFromText,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YC Text Index Switch": "Text Index Switch (YC)",
    "YC Extract Number": "Extract Number (YC)",
} 