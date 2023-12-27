
# Ship Detection In Satellite Images 
# Data Download

To use this project, you'll need to download the Airbus Ship Detection Challenge dataset from the Kaggle competition. Follow the steps below to acquire the data:

1. **Visit the Kaggle Competition Page:**
   Visit the [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection) on Kaggle to access the dataset.

2. **Download the Dataset:**
   Click on the "Data" tab on the competition page and download the necessary datasets, including training images, masks, and test images.

3. **Organize the Data:**
   Extract the downloaded data and organize it according to the directory structure expected by the script. Ensure that the training dataset includes images and corresponding segmentation masks.

4. **Configure Data Paths:**
   In the script, adjust the `ship_dir`, `train_image_dir`, and `test_image_dir` variables to point to the locations where you have stored the downloaded data.

Now you're ready to use the script for ship detection with the downloaded dataset.

---


## Overview

This repository contains a Python script for ship detection using a U-Net model with a VGG19 encoder. The U-Net architecture is a powerful tool for image segmentation tasks, and in this project, it has been tailored to identify and segment ships within images. The VGG19 model serves as the encoder, extracting valuable features for precise segmentation.

## Key Features

- **Data Processing**: The script includes comprehensive functions for reading and processing ship segmentation masks. It facilitates the creation of training and validation datasets, and it incorporates data augmentation techniques to enhance model generalization.

- **Model Architecture**: The U-Net model is constructed with a VGG19 encoder. For flexibility, the script provides options to choose from three different encoders: ResNet50, VGG19, and DenseNet121.

- **Loss Functions**: The model is compiled with a combination of binary cross-entropy and dice coefficient loss functions. This combination enhances the model's ability to accurately segment ships in diverse scenarios.

- **Training and Validation**: The script offers a user-friendly interface for training the model. Users can adjust key hyperparameters such as batch size, number of epochs, and image scaling to fine-tune the model according to their specific requirements. Training progress, losses, and metrics are visualized to assess model performance.

- **Test Predictions**: Once trained, the model can be utilized to make predictions on test images. The script provides functionality to visualize these predictions, aiding in the evaluation of the model's effectiveness.

## Usage

1. **Data Preparation**: Organize the training dataset with images and their corresponding segmentation masks. Ensure that the directory structure aligns with the script's expectations.

2. **Model Configuration**: Select the desired encoder model (VGG19, ResNet50, or DenseNet121) by uncommenting the corresponding function in the script. Additionally, configure other hyperparameters, such as batch size, epochs, and image scaling, to suit the specific requirements of your task.

3. **Training**: Execute the script to train the model. The training process will display progress, losses, and metrics. The best weights are automatically saved for future use.

4. **Test Predictions**: Leverage the trained model to make predictions on test images. The script facilitates visualization of these predictions, providing insights into the model's performance.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-image


Feel free to further customize this README to showcase any specific details or nuances of your implementation.
