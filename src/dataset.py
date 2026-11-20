
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    val_size=0.2,
    random_seed=42
):

    # Transformations applied to training images
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Transformations applied to validation and test images
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Load training and validation portion
    raw_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=False
    )

    # Get breed labels for stratified splitting
    labels = raw_dataset._labels

    indices = list(range(len(raw_dataset)))

    # Divide training data into training and validation sets
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_seed,
        stratify=labels
    )

    # Training dataset
    train_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=False
    )

    # Validation dataset
    val_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=False
    )

    # Official test dataset remains separate
    test_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        transform=eval_transform,
        download=False
    )

    # Create data loaders
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
