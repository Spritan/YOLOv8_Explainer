import os
import shutil


import cv2
import torch
import numpy as np
from PIL import Image

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression

from pytorch_grad_cam import (
    GradCAMPlusPlus,
    GradCAM,
    XGradCAM,
    EigenCAM,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    EigenGradCAM,
)

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

from .utils import letterbox


class yolov8_target:
    """
    This class is used to implement the YOLOv8 target layer.

    Args:
        ouput_type (str): The type of output to return. Can be "class", "box", or "all".
        conf (float): The confidence threshold.
        ratio (float): The ratio of maximum scores to return.

    Returns:
        A tensor containing the output.

    """

    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        """
        This function is used to perform the forward pass of the YOLOv8 target layer.

        Args:
            data (tensor): The input data.

        Returns:
            A tensor containing the output.

        """
        post_result, pre_post_boxes = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ["class", "all"]:
                result.append(post_result[i].max())
            elif self.ouput_type in ["box", "all"]:
                result.extend(pre_post_boxes[i, :4])
        return sum(result)


class yolov8_heatmap:
    """
    This class is used to implement the YOLOv8 target layer.

     Args:
            weight (str): The path to the checkpoint file.
            device (str): The device to use for inference. Defaults to "cuda:0" if a GPU is available, otherwise "cpu".
            method (str): The method to use for computing the CAM. Defaults to "EigenCAM".
            layer (list): The indices of the layers to use for computing the CAM. Defaults to [10, 12, 14, 16, 18, -3].
            backward_type (str): The type of backward pass to use. Can be "all", "positive", or "negative". Defaults to "all".
            conf_threshold (float): The confidence threshold for detections. Defaults to 0.2.
            ratio (float): The ratio of maximum scores to return. Defaults to 0.02.
            show_box (bool): Whether to show bounding boxes with the CAM. Defaults to True.
            renormalize (bool): Whether to renormalize the CAM to be in the range [0, 1] across the entire image. Defaults to False.
    
    Returns:
        A tensor containing the output.

    """

    def __init__(
        self,
        weight:str,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        method: str = "EigenCAM",
        layer=[10, 12, 14, 16, 18, -3],
        backward_type="all",
        conf_threshold=0.2,
        ratio=0.02,
        show_box=True,
        renormalize=False,
    ) -> None:
        """
        Initialize the YOLOv8 heatmap layer.
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.renormalize = renormalize
        self.show_box = show_box

        ckpt = torch.load(weight)
        self.model_names = model_names = ckpt["model"].names
        self.model = model = attempt_load_weights(weight, device)
        self.model.info()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

        self.target = target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]

        self.method = method = eval(method)(
            self.model, target_layers, use_cuda=device.type == "cuda"
        )
        method.activations_and_grads = ActivationsAndGradients( # type:ignore
            self.model, target_layers, None
        )

        self.colors = colors = np.random.uniform(
            0, 255, size=(len(model_names), 3)
        ).astype(int)
        self.__dict__.update(locals())

    def post_process(self, result):
        """
        Perform non-maximum suppression on the detections.

        Args:
            result (numpy.ndarray): The detections from the model.

        Returns:
            numpy.ndarray: The filtered detections.
        """
        return non_max_suppression(
            result, conf_thres=self.conf_threshold, iou_thres=0.65
        )[0]

    def draw_detections(self, box, color, name, img):
        """
        Draw bounding boxes and labels on an image.

        Args:
            box (list): The bounding box coordinates in the format [x1, y1, x2, y2].
            color (list): The color of the bounding box in the format [B, G, R].
            name (str): The label for the bounding box.
            img (numpy.ndarray): The image on which to draw the bounding box.

        Returns:
            numpy.ndarray: The image with the bounding box drawn.
        """
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(
            img,
            str(name),
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            tuple(int(x) for x in color),
            2,
            lineType=cv2.LINE_AA,
        )
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
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(
                grayscale_cam.shape[0] - 1, y2
            )
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(
                grayscale_cam[y1:y2, x1:x2].copy()
            )
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)  # type: ignore
        return eigencam_image_renormalized

    def renormalize_cam(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        across the entire image."""
        renormalized_cam = scale_cam_image(grayscale_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True) # type: ignore
        return eigencam_image_renormalized

    def process(self, img_path):
        """Process the input image and generate CAM visualization.

        Args:
            img_path (str): Path to the input image.
            save_path (str): Path to save the generated CAM visualization.

        Returns:
            None
        """
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0  # type: ignore

        tensor = (
            torch.from_numpy(np.transpose(img, axes=[2, 0, 1]))
            .unsqueeze(0)
            .to(self.device)
        )

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(e)
            return
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)  # type: ignore

        pred = self.model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam(
                pred[:, :4].cpu().detach().numpy().astype(np.int32), img, grayscale_cam
            )
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(
                    data[:4],
                    self.colors[int(data[4:].argmax())],
                    f"{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}",
                    cam_image,
                )

        cam_image = Image.fromarray(cam_image)
        return cam_image

    def __call__(self, img_path):
        """Generate CAM visualizations for one or more images.

        Args:
            img_path (str): Path to the input image or directory containing images.

        Returns:
            None
        """
        if os.path.isdir(img_path):
            image_list = []
            for img_path_ in os.listdir(img_path):
                img_pil = self.process(f"{img_path}/{img_path_}")
                image_list.append(img_pil)
            return image_list
        else:
            return [self.process(img_path)]
