# Comfyui-yicheng-node

A collection of image processing extension nodes for ComfyUI.

## Nodes

### Image Combine For IC

This node provides advanced image combination functions, designed for IClora and fill+redux partial redraw migration images. 
Optimize the combination of two graphs, save the size of Flux iclora and Fill+redux when partial redrawing, and save computing power.
Main features include:

1. Image Input and Masks:
   - Supports two input images with their corresponding masks
   - Mask inputs are optional, empty masks are automatically created when not connected

2. Reference Edge Selection:
   - Can select width or height from either image as reference
   - All images are scaled proportionally according to the selected reference edge
   - Options include: image1_width/image1_height/image2_width/image2_height

3. Secondary Image Control:
   - Additional scaling after reference edge scaling (0.1-2.0x)
   - 5 position placement options:
     * Vertical: top/center/bottom
     * Horizontal: left/right

4. Combination Modes:
   - Horizontal: side by side arrangement
   - Vertical: top and bottom arrangement

5. Custom Settings:
   - Adjustable final output size
   - Custom background color support (using hex color values, e.g., #FFFFFF)

## Input Parameters

1. `first_image`: First input image
2. `first_mask`: Mask for the first image (optional)
3. `second_image`: Second input image
4. `second_mask`: Mask for the second image (optional)
5. `reference_edge`: Reference edge selection
6. `combine_mode`: Combination mode (horizontal/vertical)
7. `second_image_scale`: Scaling factor for the second image (0.1-2.0)
8. `second_image_position`: Position of the second image
9. `final_size`: Final output size
10. `background_color`: Background color (e.g., #FFFFFF)

## Outputs

The node outputs:
1. `IMAGE`: Final combined image
2. `MASK`: Combined overall mask
3. `FIRST_MASK`: Separate mask for the first image (final size)
4. `SECOND_MASK`: Separate mask for the second image (final size)
5. `first_size`: Original dimensions of the first image (width, height)
6. `second_size`: Original dimensions of the second image (width, height)

## Usage Tips

1. Reference Edge Selection:
   - Choose appropriate reference edge to maintain key image proportions
   - Typically select the edge containing the main content as reference

2. Position Adjustment:
   - Use left/right for horizontal combination
   - Use top/center/bottom for vertical combination
   - Center position is ideal for alignment requirements

3. Mask Handling:
   - Masks can be left unconnected if not needed
   - Output separate masks can be used for subsequent processing or training
   - ![image](https://github.com/user-attachments/assets/c6137c53-5a6a-453a-9d38-dcf389b0b578)
   - ![image](https://github.com/user-attachments/assets/ea17cb1d-d429-47cd-b907-eb03641c186e)
   - ![image](https://github.com/user-attachments/assets/2fef7f46-9829-496f-aa58-8cdf283e4f7e)

## 关于我 | About me

Bilibili：[我的B站主页](https://space.bilibili.com/498399023?spm_id_from=333.1007.0.0)
QQ群：3260561522
wechat微信: DLONG189one



