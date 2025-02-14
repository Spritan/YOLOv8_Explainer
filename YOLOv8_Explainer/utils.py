import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.

    Args:
        im (numpy.ndarray): Input image.
        new_shape (Union[int, Tuple[int, int]], optional): Desired output shape. Defaults to (640, 640).
        color (Tuple[int, int, int], optional): Color of the border. Defaults to (114, 114, 114).
        auto (bool, optional): Whether to automatically determine padding. Defaults to True.
        scaleFill (bool, optional): Whether to stretch the image to fill the new shape. Defaults to False.
        scaleup (bool, optional): Whether to scale the image up if necessary. Defaults to True.
        stride (int, optional): Stride of the sliding window. Defaults to 32.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]: 
            - Letterboxed image
            - Ratio of resized image (width, height)
            - Padding sizes (width, height)

    Raises:
        ValueError: If input parameters are invalid
        TypeError: If input types are incorrect
    """
    # Input validation
    if not isinstance(im, np.ndarray):
        raise TypeError("Input 'im' must be a numpy array")
    
    if im.ndim != 3:
        raise ValueError("Input image must be 3-dimensional (H, W, C)")
        
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif not isinstance(new_shape, tuple) or len(new_shape) != 2:
        raise ValueError("new_shape must be an integer or tuple of two integers")
        
    if any(x <= 0 for x in new_shape):
        raise ValueError("new_shape dimensions must be positive")
        
    if not isinstance(color, tuple) or len(color) != 3:
        raise ValueError("color must be a tuple of three integers")
        
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    try:
        shape = im.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        # Divide padding into 2 sides
        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return im, ratio, (dw, dh)
        
    except Exception as e:
        raise RuntimeError(f"Error during letterbox operation: {str(e)}") from e


def display_images(images: List[Image.Image], figsize: Tuple[int, int] = (15, 7)) -> None:
    """
    Display a list of PIL images in a grid.

    Args:
        images (List[PIL.Image]): A list of PIL images to display.
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (15, 7).

    Raises:
        ValueError: If images list is empty
        TypeError: If images are not PIL.Image objects
    """
    if not images:
        raise ValueError("Images list cannot be empty")
        
    if not all(isinstance(img, Image.Image) for img in images):
        raise TypeError("All images must be PIL.Image objects")
        
    try:
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
            
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying images: {str(e)}")
    finally:
        plt.close()  # Ensure figure is closed to free memory