# Comfyui-yicheng-node

A collection of image processing extension nodes for ComfyUI.

## Nodes Description

### Image IC
Basic image combination node with the following features:
- Horizontal/vertical image concatenation
- Aspect ratio scaling based on reference edges
- Flexible position adjustment (top/center/bottom/left/right)
- Custom background color
- Mask processing support

### Image IC Advanced
Advanced image combination node, adding the following features to the basic version:
- Overlay mode support
- Precise XY coordinate control
- Independent scaling control
- Additional output options

### This two node provides advanced image combination functions, designed for IClora and fill+redux partial redraw migration images. Optimize the combination of two graphs, save the size of Flux iclora and Fill+redux when partial redrawing, and save computing power.

## How to Use

1. Drag the node into your workspace
2. Connect input images and masks (optional)
3. Set combination parameters:
   - Select reference edge
   - Set combination mode
   - Adjust position and scale
   - Set background color
4. Run the workflow to get results

## Installation

Clone this repository into ComfyUI's custom_nodes directory:
```bash
cd custom_nodes
git clone https://github.com/your-username/Comfyui-yicheng-node.git
```

   - ![image](https://github.com/user-attachments/assets/a81c8e3f-b32d-4e26-ada5-ecf145fafce6)
   - ![image](https://github.com/user-attachments/assets/a6b75f7c-d8b9-4b3e-aca6-32a6444998fb)
   - ![image](https://github.com/user-attachments/assets/26c561f3-4169-4bc6-8404-e7246349a82f)
The basic code of the nodes ic image and ic image advanced comes from https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils
Based on it, I optimized two nodes: ic image and ic image advanced.
Thanks to the original author @小志Jason

## 关于我 | About me

Bilibili：[我的B站主页](https://space.bilibili.com/498399023?spm_id_from=333.1007.0.0)
QQ号 3260561522
wechat微信: DLONG189one



