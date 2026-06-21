"""Thai Character Recognition — source package.

Modules:
    dataset  : transforms, stratified split, ThaiCharDataset, class mapping
    model    : create_model() (ResNet50 / EfficientNet-B3 / MobileNetV3), get_device()
    train    : train_model(), evaluate(), run() + CLI
    predict  : test_single_image() (inference ที่ pipeline ตรงกับตอนเทรน)
"""
