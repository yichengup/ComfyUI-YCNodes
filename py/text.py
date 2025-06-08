class YC_SingleTextNode:
    """
    单文本框编辑节点，提供文本输入和输出功能
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,  # 允许多行输入
                    "default": "text"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_text"
    CATEGORY = "YCNode/Text"

    def process_text(self, text):
        # 简单处理并返回文本
        return (text,)


class YC_FiveTextCombineNode:
    """
    五个文本框合并节点，将五个文本框的内容合并成一段连续文字，并提供单独文本输出
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text1": ("STRING", {
                    "multiline": True,
                    "default": "text1"
                }),
                "text2": ("STRING", {
                    "multiline": True,
                    "default": "text2"
                }),
                "text3": ("STRING", {
                    "multiline": True,
                    "default": "text3"
                }),
                "text4": ("STRING", {
                    "multiline": True,
                    "default": "text4"
                }),
                "text5": ("STRING", {
                    "multiline": True,
                    "default": "text5"
                }),
                "separator": ("STRING", {
                    "multiline": False,
                    "default": " "  # 默认使用空格作为分隔符，使文本连续
                })
            }
        }
    
    # 定义输出端口，每个输入文本对应一个输出端口，最后是合并的文本
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text1_out", "text2_out", "text3_out", "text4_out", "text5_out", "combined_text")
    FUNCTION = "combine_text"
    CATEGORY = "YCNode/Text"

    def combine_text(self, text1, text2, text3, text4, text5, separator):
        # 对每个文本去除首尾空白，然后使用分隔符合并成一段连续文字
        texts = [text.strip() for text in [text1, text2, text3, text4, text5] if text.strip()]
        combined = separator.join(texts)
        
        # 返回每个单独的文本和合并后的文本
        return (text1, text2, text3, text4, text5, combined)
    
class YC_textReplaceNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {
                "find1": ("STRING", {"multiline": False, "default": ""}),
                "replace1": ("STRING", {"multiline": False, "default": ""}),
                "find2": ("STRING", {"multiline": False, "default": ""}),
                "replace2": ("STRING", {"multiline": False, "default": ""}),
                "find3": ("STRING", {"multiline": False, "default": ""}),
                "replace3": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "replace_text"
    CATEGORY = "YCNode/Text"

    def replace_text(self, prompt, find1="", replace1="", find2="", replace2="", find3="", replace3=""):

        prompt = prompt.replace(find1, replace1)
        prompt = prompt.replace(find2, replace2)
        prompt = prompt.replace(find3, replace3)

        return (prompt,)

class TextKeyword:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,  # 允许多行输入
                    "default": "Your text here"
                }),
                "keyword": ("STRING", {
                    "multiline": False,  # 关键词单行输入
                    "default": "keyword"
                }),
                "mode": (["before", "after"], {
                    "default": "before"  # 默认提取关键词前的文本
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": True,
                    "label": "区分大小写"  # 是否区分大小写
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_text"
    CATEGORY = "YCNode/Text"

    def process_text(self, text, keyword, mode="before", case_sensitive=True):
        if not keyword:  # 如果关键词为空，直接返回原文本
            return (text,)
            
        if not case_sensitive:
            # 如果不区分大小写，都转换为小写进行处理
            text_to_search = text.lower()
            keyword = keyword.lower()
            # 在小写版本中找到位置
            index = text_to_search.find(keyword)
        else:
            # 区分大小写的处理
            index = text.find(keyword)
        
        # 根据模式和关键词位置处理文本
        if index == -1:  # 未找到关键词
            result = text
        elif mode == "before":
            # 返回关键词前的文本
            result = text[:index]
        else:  # mode == "after"
            # 返回关键词后的文本
            # 注意要跳过关键词本身的长度
            result = text[index + len(keyword):]
        
        # 去除结果首尾的空白字符
        result = result.strip()
        
        return (result,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YC_SingleTextNode": YC_SingleTextNode,
    "YC_FiveTextCombineNode": YC_FiveTextCombineNode,
    "YC_textReplaceNode": YC_textReplaceNode,
    "TextKeyword": TextKeyword
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "YC_SingleTextNode": "YC text box",
    "YC_FiveTextCombineNode": "YC text box combine",
    "YC_textReplaceNode": "YC text replace",
    "TextKeyword": "text keyword"
} 