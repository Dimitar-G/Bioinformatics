from torch import nn
from torchvision.models import vgg11, VGG11_Weights, efficientnet_b0, EfficientNet_B0_Weights


class EfficientClassifier(nn.Module):

    """
    A custom classifier which uses EfficientNet B0 (pretrained) for feature extraction and uses
    two fully connected layers before the final output layer.
    """

    def __init__(self):
        super(EfficientClassifier, self).__init__()
        self.efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficient.classifier[-1] = nn.Linear(1280, 320)
        self.efficient.classifier.add_module('added layer', nn.Linear(320, 1))
        self.efficient.classifier.add_module('sigmoid', nn.Sigmoid())
        # Freezing the layers for feature extraction
        for param in self.efficient.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.efficient.features(x)
        x = self.efficient.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.efficient.classifier(x)
        return x


class ALLClassifierEfficient(nn.Module):

    """
    A custom classifier which uses EfficientNet B0 (pretrained) for feature extraction and uses
    one fully connected layer before the final output layer.
    """

    def __init__(self):
        super(ALLClassifierEfficient, self).__init__()
        self.efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficient.classifier[-1] = nn.Linear(1280, 1)
        self.efficient.classifier.add_module('sigmoid', nn.Sigmoid())
        # Freezing the layers for feature extraction
        for param in self.efficient.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.efficient.features(x)
        x = self.efficient.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.efficient.classifier(x)
        return x
    

class ALLClassifier(nn.Module):

    """
    A custom classifier which uses VGG 11 (pretrained) for feature extraction and uses
    two fully connected layers before the final output layer.
    """

    def __init__(self):
        super(ALLClassifier, self).__init__()
        self.vgg = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.vgg.classifier[-2] = nn.Linear(4096, 1024)
        self.vgg.classifier[-1] = nn.Linear(1024, 1)
        self.vgg.classifier.add_module('sigmoid', nn.Sigmoid())
        # Freezing the layers for feature extraction
        for param in self.vgg.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x
