from torchvision import datasets
test_ds = datasets.ImageFolder(root='.')
print(test_ds.classes)
print(len(test_ds.classes))  # ควรได้ 72