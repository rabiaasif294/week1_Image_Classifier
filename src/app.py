
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import datasets, transforms

from model import PetCNN


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = 224
NUM_CLASSES = 37

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Load the model and class names only once
@st.cache_resource
def load_model():

    device = torch.device("cpu")

    model = PetCNN(num_classes=NUM_CLASSES).to(device)

    checkpoint_path = OUTPUT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found in outputs/best_model.pth"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get the actual class order from the Oxford-IIIT Pet dataset
    dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=False
    )

    class_names = dataset.classes

    return model, class_names


# Prepare an uploaded image
def preprocess_image(image):

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


# Streamlit page configuration
st.set_page_config(
    page_title="Pet Breed Classifier",
    page_icon="🐾",
    layout="centered"
)


# Application heading
st.title("🐾 Pet Breed Classifier")

st.write(
    "Upload a picture of a cat or dog and let the CNN "
    "predict its breed."
)

st.info("This model recognizes 37 cat and dog breeds.")


# Image uploader
uploaded_file = st.file_uploader(
    "Choose a pet image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Your uploaded pet image")

    if st.button("Predict Breed"):

        try:

            with st.spinner("Analyzing image..."):

                model, class_names = load_model()

                image_tensor = preprocess_image(image)

                with torch.no_grad():

                    outputs = model(image_tensor)

                    probabilities = torch.softmax(
                        outputs, dim=1
                    )

                    top_probabilities, top_indices = torch.topk(
                        probabilities, k=3, dim=1
                    )

            st.success("Prediction completed!")

            st.subheader("Prediction Results")

            best_index = top_indices[0][0].item()
            best_confidence = (
                top_probabilities[0][0].item() * 100
            )

            st.markdown(
                f"### Predicted Breed: {class_names[best_index]}"
            )

            st.metric(
                "Prediction Confidence",
                f"{best_confidence:.2f}%"
            )

            st.subheader("Top 3 Predictions")

            for probability, index in zip(
                top_probabilities[0], top_indices[0]
            ):

                breed = class_names[index.item()]
                confidence = probability.item() * 100

                st.write(
                    f"**{breed}** — {confidence:.2f}%"
                )

                st.progress(float(probability.item()))

        except Exception as error:

            st.error(f"Prediction failed: {error}")

else:

    st.caption("Upload an image to begin.")
