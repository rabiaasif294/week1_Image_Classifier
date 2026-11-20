
# Learning Notes — Image Classifier

## Project Goal
Build an image classification model using a Convolutional Neural Network (CNN).

## Learning Log

### Task 1 — Project Structure
- Created the required project folders.
- Learned the purpose of each directory.
- Created .gitignore to prevent datasets and model outputs from being uploaded.
- Created .dockerignore to exclude unnecessary files from Docker builds.

## Confusions and Questions
- What is the difference between .gitignore and .dockerignore?
- Why should datasets and trained model weights not be committed to GitHub?

## Future Learnings
Add new concepts, errors, solutions, and important observations here.


## Task 2 — Dataset Preparation

### Dataset Selection
- Selected the Oxford-IIIT Pet Dataset.
- Dataset contains 37 pet breed categories.
- Contains approximately 7,349 images.

### Dataset Splitting
- Used the official trainval split for training and validation.
- Divided trainval into 80% training and 20% validation.
- Used stratified splitting to preserve breed proportions.
- Kept the official test split separate for final evaluation.

### Image Preprocessing
- Resized images to 224 x 224 pixels.
- Applied random horizontal flipping to training images.
- Applied random rotation to training images.
- Converted images into PyTorch tensors.
- Applied image normalization.

### Data Loading
- Used PyTorch Dataset to access images and labels.
- Used DataLoader to load images in batches.
- Used batch size 32 as the default.
- Used separate transformations for training and evaluation.

### Important Concepts Learned
- Image preprocessing
- Data augmentation
- Normalization
- Stratified train-validation splitting
- Training, validation and test datasets
- PyTorch Dataset and DataLoader
