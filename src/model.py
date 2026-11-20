
import torch
import torch.nn as nn


class PetCNN(nn.Module):

    def __init__(self, num_classes=37):
        super(PetCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Reduce feature maps to a fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # Extract features
        x = self.features(x)

        # Reduce feature dimensions
        x = self.pool(x)

        # Predict class scores
        x = self.classifier(x)

        return x
