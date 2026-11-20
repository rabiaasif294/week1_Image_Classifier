
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



## Task 3 — CNN Model Development

### Model Architecture
- Created a custom CNN using PyTorch.
- Implemented three convolution blocks.
- Used ReLU activation after convolution layers.
- Used Max Pooling to reduce feature-map dimensions.
- Used Adaptive Average Pooling to reduce features to 1 x 1.
- Added fully connected classification layers.
- Added Dropout with probability 0.3.
- Configured the output layer for 37 pet breeds.

### Input and Output
- Input shape: Batch x 3 x 224 x 224.
- Output shape: Batch x 37.
- Output values are logits (raw class scores).

### Concepts Learned
- Convolutional Neural Networks
- Convolution filters
- ReLU activation
- Max Pooling
- Feature extraction
- Adaptive Average Pooling
- Fully Connected Layers
- Dropout
- Classification logits

## Task 5 — Model Evaluation

### Evaluation Process
- Loaded the best saved CNN checkpoint.
- Evaluated the model using the separate test dataset.
- Generated predictions for all test images.
- Compared predicted labels with actual labels.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Saved Results
- classification_report.csv
- confusion_matrix.csv
- confusion_matrix.png
- test_accuracy.txt

### Concepts Learned
- Model evaluation
- Test dataset
- Classification metrics
- True predictions
- Incorrect predictions
- Confusion matrix
- Model generalization