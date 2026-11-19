import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import ceil, sqrt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.widgets import Button
import pandas as pd

CLASS_DICT = None

def __load_class_dict():
    global CLASS_DICT

    if CLASS_DICT is not None:
        return
    
    dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.csv")

    classes_df = pd.read_csv(dict_file)
    CLASS_DICT = {
        int(row['ID']): {
            "category": row['class'],
            "IN_id": row['IN_id'],
            "WN_id": row['WN_id'],
            "dangerous": bool(int(row['dangerous']))
        }
        for _, row in classes_df.iterrows()
    }

def class_ids_to_names(class_ids, mapping_dict=None):
    if mapping_dict is None:
        __load_class_dict()
        mapping_dict = CLASS_DICT
    
    class_info = [mapping_dict[int(c)] for c in class_ids]
    class_info_keys = list(mapping_dict.values())[0].keys()

    class_info_dict = {k: [] for k in class_info_keys}
    for c in class_info:
        for k in class_info_keys:
            class_info_dict[k].append(c[k])

    return class_info_dict
    


def draw_bboxes(ax, img, bboxes, classes, name=""):
    ax.clear()  # Clear the axis before drawing
    ax.imshow(img)
    for bbox, cls in zip(bboxes, classes):
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        # Draw class name at the top-left corner of the bbox
        ax.text(
            bbox[0], bbox[1] - 5,  # Slightly above the bbox
            cls,
            color='yellow', fontsize=5, weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )
    ax.set_title(name)
    ax.axis("off")

def visualize_image(sample, fig=None, axes=None, index=None, total=None):
    """Visualizes a single synthetic sample on given matplotlib axes or creates its own figure.

    Args:
        sample (dict): A sample from the dataset.
        fig (matplotlib.figure.Figure, optional): Figure object. Created if not given.
        axes (2D list of Axes, optional): Layout [2][4]. Created if not given.
        index (int, optional): Sample index, for title.
        total (int, optional): Total number of samples, for title.
    """
    create_fig = fig is None or axes is None
    if create_fig:
        fig, axes = plt.subplots(3, 4, figsize=(18, 8))
        fig.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)
    
    xray_bboxes = sample["objects"]["xray_bbox"]

    title = f"Sample"
    if index is not None and total is not None:
        title += f" [{index}/{total}]"
    title += f" - {len(xray_bboxes)} objects"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if "categories" not in sample["objects"] and "class_IDs" in sample["objects"]:
        classes_info = class_ids_to_names(sample["objects"]["class_IDs"])
        classes = classes_info['category']
        classes_dangerous = classes_info['dangerous']

    else:
        classes = sample["objects"]["categories"]
        classes_dangerous = sample["objects"]["dangerous"]

    if "rgb_image" in sample and sample["rgb_image"] is not None:
        if "rgb_bbox" in sample["objects"] and sample["objects"]["rgb_bbox"] is not None:
            rgb_bboxes = sample["objects"]["rgb_bbox"]
        else:
            rgb_bboxes = []
        
        draw_bboxes(ax=axes[0,1], img=sample["rgb_image"], bboxes=rgb_bboxes, classes=classes, name="RGB")
        
    xray_images = [(0, "X-ray Colour", axes[0,2]), (1, "X-ray Grey", axes[0,0]), (2, "X-ray High", axes[0,3]), (3, "X-ray Low", axes[1,3]), (4, "X-ray Density", axes[1,0])]
    for i in range(5):
        im_id, im_name, im_ax = xray_images[i]
        if i < len(sample["xray_images"]) and sample["xray_images"][im_id] is not None:
            draw_bboxes(ax=im_ax, img=sample["xray_images"][im_id], bboxes=xray_bboxes, classes=classes, name=im_name)
        else:
            im_ax.axis("off")

    if "rgb_mask" in sample["objects"] and sample["objects"]["rgb_mask"] is not None:
        rgb_mask = sample["objects"]["rgb_mask"]
        create_mask_mosaic(ax=axes[1,1], masks=rgb_mask, classes=classes, name="RGB Masks")
    else:
        axes[1,1].axis("off")
    
    if "xray_mask" in sample["objects"] and sample["objects"]["xray_mask"] is not None:
        xray_mask = sample["objects"]["xray_mask"]
        create_mask_mosaic(ax=axes[1,2], masks=xray_mask, classes=classes, name="X-ray Masks")
    else:
        axes[1,2].axis("off")

    # Add object information to axes[2,2] (odd objects) and axes[2,3] (even objects)
    axes[2,1].clear()
    axes[2,1].axis('off')
    axes[2,2].clear()
    axes[2,2].axis('off')
    
    if classes and len(xray_bboxes) > 0:
        # Get object properties from sample
        dangerous = classes_dangerous
        rgb_visible = sample.get("rgb_visible", [False] * len(classes))
        xray_visible = sample.get("xray_visible", [False] * len(classes))
        
        # Ensure lists are the same length as classes
        if not isinstance(dangerous, list):
            dangerous = [dangerous] * len(classes)
        if not isinstance(rgb_visible, list):
            rgb_visible = [rgb_visible] * len(classes)
        if not isinstance(xray_visible, list):
            xray_visible = [xray_visible] * len(classes)
        
        left_info = "Class name, Danger., Xray vis., RGB vis.\n\n"
        right_info = "Class name, Danger., Xray vis., RGB vis.\n\n"
        
        for i, (cls, dang, rgb_vis, xray_vis) in enumerate(zip(classes, dangerous, rgb_visible, xray_visible)):
            # Format the object information with true/false values
            object_line = f"{cls}, {str(dang).lower()}, {str(xray_vis).lower()}, {str(rgb_vis).lower()}\n"
            
            # Distribute objects: odd indices to left, even to right
            if i % 2 == 0:
                left_info += object_line
            else:  # Odd indices go to left
                right_info += object_line
        
        # Display on respective axes
        axes[2,1].text(0.05, 0.95, left_info, transform=axes[2,1].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        axes[2,2].text(0.05, 0.95, right_info, transform=axes[2,2].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    else:
        no_info = "No objects available"
        axes[2,1].text(0.05, 0.95, no_info, transform=axes[2,1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[2,2].text(0.05, 0.95, no_info, transform=axes[2,2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    if create_fig:
        plt.show()
    else:
        fig.canvas.draw_idle()

def _convert_to_pil_image(img):
    """Convert HuggingFace image dictionary or numpy array to PIL Image."""
    # Check if already a PIL Image first - no conversion needed
    if isinstance(img, Image.Image):
        return img
    
    # Only convert if not already PIL
    if isinstance(img, dict):
        # HuggingFace format: dictionary with 'bytes' or 'path' key
        if 'bytes' in img and img['bytes'] is not None:
            from io import BytesIO
            return Image.open(BytesIO(img['bytes']))
        elif 'path' in img and img['path'] is not None:
            return Image.open(img['path'])
        else:
            raise ValueError("Image dictionary must have 'bytes' or 'path' key")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img)
    else:
        # Try to convert to numpy array first
        return Image.fromarray(np.array(img))

def create_mask_mosaic(ax, masks, classes=None, name=""):
    ax.clear()  # Clear the axis before drawing
    num_masks = len(masks)

    ax.set_title(name)
    ax.axis('off')

    if len(masks) == 0:
        return

    # Convert first mask to PIL Image to get size
    first_mask = _convert_to_pil_image(masks[0])
    size = first_mask.size

    grid_size = ceil(sqrt(num_masks))

    # Create empty mosaic image
    mosaic = Image.new('L', (grid_size * size[0], grid_size * size[1]))

    # Optional: for drawing labels on top
    mosaic_rgb = Image.new('RGB', mosaic.size)
    draw = ImageDraw.Draw(mosaic_rgb)

    # Handle case where classes might be None or have different length
    if classes is None:
        classes = [f"Mask {i}" for i in range(num_masks)]
    elif len(classes) != num_masks:
        # Pad or truncate classes to match masks length
        classes = list(classes[:num_masks]) + [f"Mask {i}" for i in range(len(classes), num_masks)]
    
    for idx, (mask, label) in enumerate(zip(masks, classes)):
        # Convert mask to PIL Image if needed
        mask_pil = _convert_to_pil_image(mask)
        row, col = divmod(idx, grid_size)
        x, y = col * size[0], row * size[1]
        mosaic_rgb.paste(mask_pil, (x, y))
        draw.rectangle([x, y, x + size[0], y + size[1]], outline='white', width=1)
        ax.text(
            x + 3, y + 3,  # Slightly above the bbox
            label,
            color='yellow', fontsize=5, weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )

    ax.imshow(mosaic_rgb)



def visualize_crop(sample, fig=None, axes=None, index=None, total=None):
    """Visualizes a single crop sample on given matplotlib axes or creates its own figure.

    Args:
        sample (dict): A sample from the dataset.
        fig (matplotlib.figure.Figure, optional): Figure object. Created if not given.
        axes (2D list of Axes, optional): Layout [3][5]. Created if not given.
        index (int, optional): Sample index, for title.
        total (int, optional): Total number of samples, for title.
    """
    create_fig = fig is None or axes is None
    if create_fig:
        fig, axes = plt.subplots(3, 5, figsize=(18, 8))
        fig.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)

    title = f"Sample"
    if index is not None and total is not None:
        title += f" [{index}/{total}]"
    title += f" - Category: {sample['category']}, IN_id: {sample['IN_id']}, WN_id: {sample['WN_id']}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    metadata_text = (
        f"Category: {sample['category']}\n"
        f"Dangerous: {'Yes' if sample['dangerous'] else 'No'}\n"
        f"Is Background: {'Yes' if sample['is_background'] else 'No'}\n"
        f"RGB Visible: {'Yes' if sample['rgb_visible'] else 'No'}\n"
        f"X-ray Visible: {'Yes' if sample['xray_visible'] else 'No'}\n"
        f"RGB Size: {sample['rgb_width']}x{sample['rgb_height']}\n"
        f"X-ray Size: {sample['xray_width']}x{sample['xray_height']}"
    )

    axes[2][2].cla()
    axes[2][2].axis('off')
    axes[2][2].text(0, 0.5, metadata_text, fontsize=10, verticalalignment='center')

    def show(ax, img, title, cmap=None):
        ax.cla()
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    show(axes[0][0], sample['rgb_image'], "RGB Image")
    show(axes[0][1], sample['rgb_image_mask'], "RGB Mask", cmap='gray')
    show(axes[0][2], sample['xray_image_colour'], "X-ray Colour")
    show(axes[0][3], sample['xray_image_highlow'], "X-ray High/Low")
    show(axes[0][4], sample['xray_image_mask'], "X-ray Mask", cmap='gray')

    show(axes[1][0], sample['raw_xray_image_colour'], "Raw X-ray Colour")
    show(axes[1][1], sample['raw_xray_image_grey'], "Raw X-ray Grey")
    show(axes[1][2], sample['raw_xray_image_high'], "Raw X-ray High", cmap='gray')
    show(axes[1][3], sample['raw_xray_image_low'], "Raw X-ray Low", cmap='gray')
    show(axes[1][4], sample['raw_xray_image_density'], "Raw X-ray Density", cmap='gray')

    for ax in [axes[2][1], axes[2][3]]:
        ax.cla()
        ax.axis('off')

    if create_fig:
        axes[2][0].axis('off')
        axes[2][4].axis('off')
        plt.show()
    else:
        fig.canvas.draw_idle()


def launch_dataset_viewer(dataset, initial_index=0):
    index = [initial_index]

    fig, axes = plt.subplots(3, 5, figsize=(18, 8))
    fig.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)

    for ax in [axes[2][0], axes[2][4]]:
        ax.axis('off')
    button_ax_prev = axes[2][0]
    button_ax_next = axes[2][4]
    btn_prev = Button(button_ax_prev, '◀ Prev')
    btn_next = Button(button_ax_next, 'Next ▶')

    def update_sample(delta):
        index[0] = (index[0] + delta) % len(dataset)
        visualize_crop(dataset[index[0]], fig, axes, index[0], len(dataset))

    def on_key(event):
        if event.key in ['right', 'd']:
            update_sample(+1)
        elif event.key in ['left', 'a']:
            update_sample(-1)

    btn_prev.on_clicked(lambda event: update_sample(-1))
    btn_next.on_clicked(lambda event: update_sample(+1))
    fig.canvas.mpl_connect('key_press_event', on_key)

    visualize_crop(dataset[index[0]], fig, axes, index[0], len(dataset))
    plt.show()


def launch_dataset_viewer_synthetic(dataset, initial_index=0):
    """Launch a dataset viewer for synthetic data using visualize_image().
    
    Args:
        dataset: Dataset to visualize
        initial_index: Starting index for the viewer
    """
    index = [initial_index]
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 8))
    fig.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)
    
    # Add navigation buttons
    for ax in [axes[2][0], axes[2][3]]:
        ax.axis('off')
    button_ax_prev = axes[2][0]
    button_ax_next = axes[2][3]
    btn_prev = Button(button_ax_prev, '◀ Prev')
    btn_next = Button(button_ax_next, 'Next ▶')
    
    def update_sample(delta):
        index[0] = (index[0] + delta) % len(dataset)
        visualize_image(dataset[index[0]], fig, axes, index[0], len(dataset))
        
    def on_key(event):
        if event.key in ['right', 'd']:
            update_sample(+1)
        elif event.key in ['left', 'a']:
            update_sample(-1)
    
    btn_prev.on_clicked(lambda event: update_sample(-1))
    btn_next.on_clicked(lambda event: update_sample(+1))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    visualize_image(dataset[index[0]], fig, axes, index[0], len(dataset))
    plt.show()
