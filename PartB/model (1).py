import torch.nn as nn
from torchvision import models

# Function to load a pretrained model, freeze some layers, and modify the final layer for classification
def load_model(model_name="resnet50", num_classes=10, freeze_percentage=1.0):
    
    # Load the specified pretrained model with ImageNet weights
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Decide how many parameters to freeze
    target_freeze = int(total_params * freeze_percentage)
    frozen = 0

    # Freeze the required number of parameters
    for param in model.parameters():
        if frozen >= target_freeze:
            break
        param.requires_grad = False
        frozen += param.numel()

    # Replace the final classification layer to match the number of classes
    if model_name in ["resnet50", "resnet18", "googlenet", "inception_v3"]:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Also update the auxiliary classifier in Inception V3 if present
    if model_name == "inception_v3" and model.aux_logits:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    # Print information about total and frozen parameters
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen:,} ({100*frozen/total_params:.1f}%)")

    # Return the modified model
    return model
