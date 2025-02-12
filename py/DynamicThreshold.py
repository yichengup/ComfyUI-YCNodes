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
    CATEGORY = "YCNode/utils"

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

NODE_CLASS_MAPPINGS = {
    # ... other nodes
    "DynamicThreshold": DynamicThreshold
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ... other nodes
    "DynamicThreshold": "Dynamic Threshold" 
}
