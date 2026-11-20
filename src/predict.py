
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

from src.model import PetCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = 224
NUM_CLASSES = 37

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Oxford-IIIT Pet class names
CLASS_NAMES = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier"
]


def predict_image(image_path):

    device = torch.device("cpu")

    # Check whether image exists
    image_path = Path(image_path)

    if not image_path.exists():
        print("Error: Image file not found.")
        return

    # Prepare image transformation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Open and prepare image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Load trained model
    model = PetCNN(num_classes=NUM_CLASSES).to(device)

    checkpoint_path = OUTPUT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        print("Error: Trained model checkpoint not found.")
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict breed
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    class_id = predicted_class.item()
    confidence_value = confidence.item() * 100

    print("\nPrediction Results")
    print("------------------")
    print("Image:", image_path.name)
    print("Predicted breed:", CLASS_NAMES[class_id])
    print(f"Confidence: {confidence_value:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide an image path.")
        print(
            'Example: python -m src.predict "data/my_pet.jpg"'
        )
    else:
        predict_image(sys.argv[1])
