# YOLOv8_Explainer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://static.pepy.tech/personalized-badge/grad-cam?period=month&units=international_system&left_color=black&right_color=brightgreen&left_text=Monthly%20Downloads)](https://pepy.tech/project/YOLOv8-Explainer)
[![Downloads](https://static.pepy.tech/personalized-badge/grad-cam?period=total&units=international_system&left_color=black&right_color=blue&left_text=Total%20Downloads)](https://pepy.tech/project/YOLOv8-Explainer) -->
## Simplify your understanding of YOLOv8 Results
This is a package with state of the art methods for Explainable AI for computer vision using YOLOv8. This can be used for diagnosing model predictions, either in production or while developing models. The aim is also to serve as a benchmark of algorithms and metrics for research of new explainability methods.

### Install Environment & Dependencies

`YOLOv8-Explainer` can be seamlessly integrated into your projects with a straightforward installation process:

#### Installation as a Package

To incorporate `YOLOv8-Explainer` into your project as a dependency, execute the following command in your terminal:

```bash
pip install YOLOv8-Explainer
```

## Features and Functionality

`YOLOv8-Explainer`  can be used to deploy various different CAM models for cutting-edge XAI methodologies in `YOLOv8` for images:

- GradCAM : Weight the 2D activations by the average gradient
- GradCAM + + : Like GradCAM but uses second order gradients
- XGradCAM : Like GradCAM but scale the gradients by the normalized activations
- EigenCAM : Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results)
- HiResCAM : Like GradCAM but element-wise multiply the activations with the gradients; provably guaranteed faithfulness for certain models
- LayerCAM : Spatially weight the activations by positive gradients. Works better especially in lower layers
- EigenGradCAM : Like EigenCAM but with class discrimination: First principle component of Activations*Grad. Looks like GradCAM, but cleaner


# Using from code as a library

```python

from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="/location/model.pt", 
        conf_threshold=0.4, 
        device = "cpu", 
        method = "EigenCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        backward_type="all",
        ratio=0.02,
        show_box=True,
        renormalize=False,
)

imagelist = model(
    img_path="/location/image.jpg", 
    )

display_images(imagelist)

```

You can choose between the following CAM Models for version 0.0.2:

`GradCAM` , `HiResCAM`, `GradCAMPlusPlus`, `XGradCAM` , `LayerCAM`, `EigenGradCAM` and `EigenCAM`.

You can add a single image or a directory images to be used by the `Module`. The output will be a corresponding list of images (list contianing one PIL Image for a single image imput and list contining as many PIL images as Images in the input directory).


# References
https://github.com/jacobgil/pytorch-grad-cam <br>
`PyTorch library for CAM methods
Jacob Gildenblat and contributors`

https://github.com/z1069614715/objectdetection_script <br>
`Object Detection Script
Devil's Mask`

https://arxiv.org/abs/1610.02391 <br>
`Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra`

https://arxiv.org/abs/2011.08891 <br>
`Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks
Rachel L. Draelos, Lawrence Carin`

https://arxiv.org/abs/1710.11063 <br>
`Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian`

https://arxiv.org/abs/1910.01279 <br>
`Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, Xia Hu`

https://ieeexplore.ieee.org/abstract/document/9093360/ <br>
`Ablation-cam: Visual explanations for deep convolutional network via gradient-free localization.
Saurabh Desai and Harish G Ramaswamy. In WACV, pages 972–980, 2020`

https://arxiv.org/abs/2008.02312 <br>
`Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs
Ruigang Fu, Qingyong Hu, Xiaohu Dong, Yulan Guo, Yinghui Gao, Biao Li`

https://arxiv.org/abs/2008.00299 <br>
`Eigen-CAM: Class Activation Map using Principal Components
Mohammed Bany Muhammad, Mohammed Yeasin`

http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf <br>
`LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
Peng-Tao Jiang; Chang-Bin Zhang; Qibin Hou; Ming-Ming Cheng; Yunchao Wei`

https://arxiv.org/abs/1905.00780 <br>
`Full-Gradient Representation for Neural Network Visualization
Suraj Srinivas, Francois Fleuret`

https://arxiv.org/abs/1806.10206 <br>
`Deep Feature Factorization For Concept Discovery
Edo Collins, Radhakrishna Achanta, Sabine Süsstrunk`