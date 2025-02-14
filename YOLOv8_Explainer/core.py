import os
import shutil
from typing import List, Optional, Tuple, Union
from threading import Lock
import gc

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus,
                              HiResCAM, LayerCAM, RandomCAM, XGradCAM)
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

from .utils import letterbox


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 reshape_transform: Optional[callable]) -> None:  # type: ignore
        """
        Initializes the ActivationsAndGradients object.

        Args:
            model (torch.nn.Module): The neural network model.
            target_layers (List[torch.nn.Module]): List of target layers from which to extract activations and gradients.
            reshape_transform (Optional[callable]): A function to transform the shape of the activations and gradients if needed.

        Raises:
            ValueError: If model or target_layers is None
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if not target_layers:
            raise ValueError("Target layers cannot be empty")

        self.model = model
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.reshape_transform = reshape_transform
        self.handles: List = []
        self._lock = Lock()
        
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module: torch.nn.Module,
                        input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                        output: torch.Tensor) -> None:
        """
        Saves the activation of the targeted layer.

        Args:
            module (torch.nn.Module): The targeted layer module.
            input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): The input to the targeted layer.
            output (torch.Tensor): The output activation of the targeted layer.
        """
        with self._lock:
            try:
                activation = output.clone()
                if self.reshape_transform is not None:
                    activation = self.reshape_transform(activation)
                self.activations.append(activation)
            except Exception as e:
                print(f"Error saving activation: {str(e)}")

    def save_gradient(self, module: torch.nn.Module,
                     grad_input: Tuple[torch.Tensor, ...],
                     grad_output: Tuple[torch.Tensor, ...]) -> None:
        """
        Saves the gradient of the targeted layer.

        Args:
            module (torch.nn.Module): The targeted layer module.
            grad_input (Tuple[torch.Tensor, ...]): The input gradients.
            grad_output (Tuple[torch.Tensor, ...]): The output gradients.
        """
        with self._lock:
            try:
                if len(grad_output) > 0:
                    grad = grad_output[0].clone()
                    if self.reshape_transform is not None:
                        grad = self.reshape_transform(grad)
                    self.gradients = [grad] + self.gradients
            except Exception as e:
                print(f"Error saving gradient: {str(e)}")

    def post_process(self, result: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Post-processes the result.

        Args:
            result (torch.Tensor): The result tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray]: A tuple containing the post-processed result.
        """
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x: torch.Tensor) -> List[List[Union[torch.Tensor, np.ndarray]]]:
        """
        Calls the ActivationsAndGradients object.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            List[List[Union[torch.Tensor, np.ndarray]]]: A list containing activations and gradients.
        """
        try:
            with self._lock:
                self.gradients = []
                self.activations = []
            
            # Forward pass
            self.model.zero_grad()
            model_output = self.model(x)
            
            if not isinstance(model_output, (tuple, list)):
                model_output = [model_output]
            
            # Process output and compute loss for backward pass
            post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
            
            # Ensure we have a scalar for backward pass
            if len(post_result) > 0:
                loss = post_result.sum()
                # Backward pass
                loss.backward(retain_graph=True)
            
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return [[post_result, pre_post_boxes]]
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise
        finally:
            gc.collect()  # Force garbage collection

    def release(self) -> None:
        """Removes hooks and cleans up resources."""
        try:
            for handle in self.handles:
                if handle is not None:
                    handle.remove()
            self.handles.clear()
            self.gradients.clear()
            self.activations.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error releasing resources: {str(e)}")
        finally:
            gc.collect()

    def __del__(self):
        """Destructor to ensure resource cleanup."""
        self.release()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in range(post_result.size(0)):
            if float(post_result[i].max()) >= self.conf:
                if self.ouput_type == 'class' or self.ouput_type == 'all':
                    result.append(post_result[i].max())
                if self.ouput_type == 'box' or self.ouput_type == 'all':
                    for j in range(4):
                        result.append(pre_post_boxes[i, j])
        return sum(result)


class yolov8_heatmap:
    """
    This class implements various CAM visualization methods for YOLOv8 models.

    Args:
        weight (str): Path to the model weights file.
        device (str, optional): Device to run the model on. Defaults to CUDA if available.
        method (str, optional): CAM method to use. Defaults to "EigenGradCAM".
        layer (list, optional): Layer indices for visualization. Defaults to [12, 17, 21].
        conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.2.
        ratio (float, optional): Ratio for maximum scores. Defaults to 0.02.
        show_box (bool, optional): Whether to show bounding boxes. Defaults to True.
        renormalize (bool, optional): Whether to renormalize CAM. Defaults to False.

    Raises:
        ValueError: If invalid parameters are provided
        FileNotFoundError: If weight file doesn't exist
        RuntimeError: If model loading fails
    """

    VALID_METHODS = {
        "GradCAM", "GradCAMPlusPlus", "XGradCAM", "EigenCAM",
        "EigenGradCAM", "LayerCAM", "HiResCAM"
    }

    # Define default target layers for different YOLOv8 sizes
    DEFAULT_TARGET_LAYERS = {
        'n': [12, 17, 21],  # nano
        's': [15, 21, 27],  # small
        'm': [18, 25, 32],  # medium
        'l': [21, 29, 37],  # large
        'x': [24, 33, 42],  # xlarge
    }

    def __init__(
            self,
            weight: str,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            method="EigenGradCAM",
            layer=None,  # Now optional
            conf_threshold=0.2,
            ratio=0.02,
            show_box=True,
            renormalize=False,
    ) -> None:
        """Initialize the YOLOv8 heatmap generator."""
        try:
            # Validate inputs
            if not os.path.exists(weight):
                raise FileNotFoundError(f"Weight file not found: {weight}")
            
            if method not in self.VALID_METHODS:
                raise ValueError(f"Invalid method {method}. Must be one of {self.VALID_METHODS}")
            
            # Load model
            self.device = device
            ckpt = torch.load(weight, map_location=device)
            self.model_names = ckpt['model'].names
            self.model = attempt_load_weights(weight, device)
            
            # Determine model size and appropriate layers
            if layer is None:
                # Try to determine model size from weight file name
                model_size = 'n'  # default to nano
                for size in self.DEFAULT_TARGET_LAYERS.keys():
                    if f"yolov8{size}" in weight.lower():
                        model_size = size
                        break
                layer = self.DEFAULT_TARGET_LAYERS[model_size]
            
            if not isinstance(layer, (list, tuple)) or not all(isinstance(l, int) for l in layer):
                raise ValueError("Layer must be a list of integers")
            
            if not 0 <= conf_threshold <= 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
            
            if not 0 <= ratio <= 1:
                raise ValueError("Ratio must be between 0 and 1")

            # Enable gradients for all parameters
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.eval()

            # Initialize target and layers
            self.target = yolov8_target("all", conf_threshold, ratio)
            
            # Safely get target layers
            try:
                self.target_layers = []
                for l in layer:
                    try:
                        target_layer = self.model.model[l]
                        # Verify it's a valid layer type that can produce activations
                        if hasattr(target_layer, 'forward'):
                            self.target_layers.append(target_layer)
                        else:
                            print(f"Warning: Layer at index {l} is not a valid target layer, skipping")
                    except IndexError:
                        print(f"Warning: Layer index {l} is out of range, skipping")
                
                if not self.target_layers:
                    raise ValueError("No valid target layers found")
                
            except Exception as e:
                raise ValueError(f"Error accessing model layers: {str(e)}. Model structure may have changed.") from e

            # Initialize CAM method
            try:
                cam_class = eval(method)
                self.method = cam_class(
                    model=self.model,
                    target_layers=self.target_layers,
                    use_cuda=device.type == 'cuda'
                )
                self.method.activations_and_grads = ActivationsAndGradients(
                    self.model, self.target_layers, None
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CAM method: {str(e)}") from e

            # Store configuration
            self.conf_threshold = conf_threshold
            self.show_box = show_box
            self.renormalize = renormalize
            self.colors = np.random.uniform(0, 255, size=(len(self.model_names), 3)).astype(int)

        except Exception as e:
            # Clean up resources in case of initialization failure
            self.release()
            raise RuntimeError(f"Failed to initialize YOLOv8 heatmap: {str(e)}") from e

    def post_process(self, result: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Process model output with non-maximum suppression.

        Args:
            result (torch.Tensor): Raw model output

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray]: Processed detections
        """
        try:
            # Move computation to appropriate device
            result = result.to(self.device)
            
            # Apply NMS
            processed_result = non_max_suppression(
                result,
                conf_thres=self.conf_threshold,
                iou_thres=0.45
            )

            if len(processed_result) == 0 or processed_result[0].numel() == 0:
                return torch.empty(0, 6, device=self.device), torch.empty(0, 4, device=self.device), np.array([])

            # Get first batch results
            detections = processed_result[0]
            
            # Filter by confidence
            mask = detections[:, 4] >= self.conf_threshold
            filtered_detections = detections[mask]
            
            # Sort by confidence
            sorted_indices = torch.argsort(filtered_detections[:, 4], descending=True)
            sorted_detections = filtered_detections[sorted_indices]
            
            # Extract logits and boxes
            logits = sorted_detections[:, 4:]
            boxes = sorted_detections[:, :4]
            
            return logits, boxes, xywh2xyxy(boxes).cpu().numpy()

        except Exception as e:
            print(f"Error in post-processing: {str(e)}")
            return torch.empty(0, 6, device=self.device), torch.empty(0, 4, device=self.device), np.array([])

    def release(self) -> None:
        """Release resources and clean up memory."""
        try:
            if hasattr(self, 'method') and hasattr(self.method, 'activations_and_grads'):
                self.method.activations_and_grads.release()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error releasing resources: {str(e)}")
        finally:
            gc.collect()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.release()

    def draw_detections(self, box, color, name, img):
        """
        Draw bounding boxes and labels on an image for multiple detections.

        Args:
            box (torch.Tensor or np.ndarray): The bounding box coordinates in the format [x1, y1, x2, y2]
            color (list): The color of the bounding box in the format [B, G, R]
            name (str): The label for the bounding box.
            img (np.ndarray): The image on which to draw the bounding box

        Returns:
            np.ndarray: The image with the bounding box drawn.
        """
        # Ensure box coordinates are integers
        xmin, ymin, xmax, ymax = map(int, box[:4])

        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      tuple(int(x) for x in color), 2)

        # Draw label
        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)

        return img

    def renormalize_cam_in_bounding_boxes(
            self,
            boxes: np.ndarray,  # type: ignore
            image_float_np: np.ndarray,  # type: ignore
            grayscale_cam: np.ndarray,  # type: ignore
    ) -> np.ndarray:
        """
        Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes.

        Args:
            boxes (np.ndarray): The bounding boxes.
            image_float_np (np.ndarray): The image as a numpy array of floats in the range [0, 1].
            grayscale_cam (np.ndarray): The CAM as a numpy array of floats in the range [0, 1].

        Returns:
            np.ndarray: The renormalized CAM.
        """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1,
                         x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(
                grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(
            image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def renormalize_cam(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        across the entire image."""
        renormalized_cam = scale_cam_image(grayscale_cam)
        eigencam_image_renormalized = show_cam_on_image(
            image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path: str) -> Optional[Image.Image]:
        """
        Process the input image and generate CAM visualization.
        
        Args:
            img_path (str): Path to the input image
            
        Returns:
            Optional[Image.Image]: Processed image with CAM visualization, or None if processing fails
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If processing fails
        """
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            # Letterbox and convert to RGB
            img = letterbox(img)[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0

            # Convert to tensor and ensure correct device
            tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0)
            tensor = tensor.to(self.device, non_blocking=True)
            tensor.requires_grad = True  # Enable gradients for input

            try:
                # Generate CAM
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    # Clear any existing gradients
                    self.model.zero_grad()
                    if hasattr(self.method, 'activations_and_grads'):
                        self.method.activations_and_grads.gradients = []
                        self.method.activations_and_grads.activations = []
                    
                    # Generate CAM
                    grayscale_cam = self.method(tensor, [self.target])
                    if grayscale_cam is None:
                        raise RuntimeError("CAM generation failed")
                    grayscale_cam = grayscale_cam[0, :]

                    # Get predictions
                    with torch.no_grad():  # No need for gradients in detection
                        pred1 = self.model(tensor)[0]
                        pred = non_max_suppression(
                            pred1,
                            conf_thres=self.conf_threshold,
                            iou_thres=0.45
                        )[0]

            except Exception as e:
                raise RuntimeError(f"Error during model inference: {str(e)}") from e

            finally:
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Process CAM visualization
            try:
                if self.renormalize and len(pred) > 0:
                    cam_image = self.renormalize_cam(
                        pred[:, :4].cpu().numpy().astype(np.int32),
                        img,
                        grayscale_cam
                    )
                else:
                    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # Draw bounding boxes if requested
                if self.show_box and len(pred) > 0:
                    pred_cpu = pred.cpu().numpy()
                    for detection in pred_cpu:
                        # Get class index and confidence
                        class_index = int(detection[5])
                        conf = detection[4]

                        # Draw detection
                        cam_image = self.draw_detections(
                            detection[:4],  # Box coordinates
                            self.colors[class_index],  # Color for this class
                            f"{self.model_names[class_index]} {conf:.2f}",  # Label with confidence
                            cam_image,
                        )

                return Image.fromarray(cam_image)

            except Exception as e:
                raise RuntimeError(f"Error during visualization: {str(e)}") from e

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

    def __call__(self, img_path: Union[str, List[str]]) -> List[Optional[Image.Image]]:
        """
        Generate CAM visualizations for one or more images.

        Args:
            img_path (Union[str, List[str]]): Path to input image(s) or directory

        Returns:
            List[Optional[Image.Image]]: List of processed images, None for failed items
        """
        try:
            # Handle directory input
            if os.path.isdir(img_path):
                image_paths = [
                    os.path.join(img_path, f) 
                    for f in os.listdir(img_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ]
            # Handle single image or list of images
            elif isinstance(img_path, str):
                image_paths = [img_path]
            elif isinstance(img_path, (list, tuple)):
                image_paths = img_path
            else:
                raise ValueError("img_path must be a string path or list of paths")

            results = []
            total = len(image_paths)

            for i, path in enumerate(image_paths, 1):
                try:
                    print(f"Processing image {i}/{total}: {path}")
                    result = self.process(path)
                    results.append(result)
                    
                except Exception as e:
                    print(f"Failed to process {path}: {str(e)}")
                    results.append(None)
                
                finally:
                    # Ensure memory is cleared after each image
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

            return results

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return []
