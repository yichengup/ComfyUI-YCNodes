class TextBeforeKeyword:
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

    def process_text(self, text, keyword, case_sensitive=True):
        if not case_sensitive:
            # 如果不区分大小写，都转换为小写进行处理
            text_to_search = text.lower()
            keyword = keyword.lower()
            # 在小写版本中找到位置
            index = text_to_search.find(keyword)
            # 但返回原始文本的对应部分
            result = text[:index] if index != -1 else text
        else:
            # 区分大小写的处理
            index = text.find(keyword)
            result = text[:index] if index != -1 else text
        
        # 去除结果末尾的空白字符
        result = result.rstrip()
        
        return (result,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "TextBeforeKeyword": TextBeforeKeyword
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBeforeKeyword": "Text Before Keyword"
} 
