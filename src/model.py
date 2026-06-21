"""Model factory for Thai Character Recognition (Transfer Learning).

โมเดลทั้ง 3 ถูกแก้ Conv layer แรกให้รับภาพ grayscale 1 channel
และเปลี่ยนหัว classifier ให้มี num_classes ตามชุดข้อมูล
"""
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ["resnet50", "efficientnet_b3", "mobilenet_v3"]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "efficientnet_b3": "EfficientNet-B3",
    "mobilenet_v3": "MobileNetV3-Large",
}


def create_model(model_name, num_classes, pretrained=True, device=None):
    """สร้างโมเดล transfer learning สำหรับ grayscale input

    Parameters
    ----------
    pretrained : bool
        True = โหลด ImageNet weights (ตอนเทรน), False = สุ่ม weights
        (ตอน inference ที่จะ load_state_dict ทับอยู่แล้ว)
    """
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False,
        )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")

    if device is not None:
        model = model.to(device)
    return model


def get_device():
    """เลือก device ที่ดีที่สุด: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
