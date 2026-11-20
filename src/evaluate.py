
from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.dataset import get_dataloaders
from src.model import PetCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 32
NUM_CLASSES = 37


def evaluate_model():

    device = torch.device("cpu")

    # Load the test dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("Test images:", len(test_loader.dataset))

    # Create the model
    model = PetCNN(num_classes=NUM_CLASSES).to(device)

    # Load the best trained checkpoint
    checkpoint_path = OUTPUT_DIR / "best_model.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best checkpoint loaded.")
    print("Checkpoint epoch:", checkpoint["epoch"])

    all_predictions = []
    all_labels = []

    # Evaluate using test images
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate test accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

    # Generate classification report
    class_names = [
        f"Class_{i}" for i in range(NUM_CLASSES)
    ]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    # Save classification report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "classification_report.csv"

    with open(report_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Class",
            "Precision",
            "Recall",
            "F1-Score",
            "Support"
        ])

        for class_name, values in report.items():

            if isinstance(values, dict):

                writer.writerow([
                    class_name,
                    values["precision"],
                    values["recall"],
                    values["f1-score"],
                    values["support"]
                ])

    # Save confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    cm_path = RESULTS_DIR / "confusion_matrix.csv"

    with open(cm_path, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerows(cm)

    # Plot confusion matrix
    plt.figure(figsize=(16, 14))

    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.colorbar()

    plt.xticks(range(NUM_CLASSES), range(NUM_CLASSES), rotation=90)
    plt.yticks(range(NUM_CLASSES), range(NUM_CLASSES))

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # Save overall accuracy
    with open(RESULTS_DIR / "test_accuracy.txt", "w") as file:

        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Test Images: {len(all_labels)}\n")

    print("\nEvaluation completed!")
    print("Classification report:", report_path)
    print("Confusion matrix:", cm_path)
    print("Confusion matrix plot:", RESULTS_DIR / "confusion_matrix.png")
    print("Accuracy file:", RESULTS_DIR / "test_accuracy.txt")


if __name__ == "__main__":
    evaluate_model()
