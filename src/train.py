
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import PetCNN


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training settings
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Record loss and accuracy
        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def save_results(history):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save training history as a CSV file
    csv_path = RESULTS_DIR / "training_history.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    # Plot training and validation loss
    epochs = [item["epoch"] for item in history]

    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, marker="o", label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

    # Plot training and validation accuracy
    train_acc = [item["train_accuracy"] for item in history]
    val_acc = [item["val_accuracy"] for item in history]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
    plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(RESULTS_DIR / "accuracy_curve.png")
    plt.close()

    print("\nTraining history saved.")
    print("CSV:", csv_path)
    print("Loss graph:", RESULTS_DIR / "loss_curve.png")
    print("Accuracy graph:", RESULTS_DIR / "accuracy_curve.png")


def main():

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Use CPU for this baseline experiment
    device = torch.device("cpu")

    print("Using device:", device)

    # Create required folders
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    # Create CNN
    model = PetCNN(num_classes=37).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []
    best_val_loss = float("inf")

    print("\nStarting model training...")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # Store epoch results
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

        history.append(epoch_results)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                OUTPUT_DIR / "best_model.pth"
            )

            print("Best model saved!")

    # Save training results
    save_results(history)

    print("\nTraining completed!")
    print("Best validation loss:", round(best_val_loss, 4))
    print("Best model:", OUTPUT_DIR / "best_model.pth")


if __name__ == "__main__":
    main()
