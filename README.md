# Marine-Life
# Object Detection, Image Classification, and Few-Shot Learning with YOLOv7, CNNs, and Siamese Networks

This project combines object detection using the YOLOv7 model, image classification using convolutional neural networks (CNNs), and few-shot learning using Siamese networks. The code performs various tasks, including preprocessing the dataset, training the models, and visualizing the training progress.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Tasks](#2-tasks)
    - [2.1 Image Classification with CNNs](#21-image-classification-with-cnns)
        - [2.1.1 Data Preparation and Preprocessing](#211-data-preparation-and-preprocessing)
        - [2.1.2 Model Architecture and Training](#212-model-architecture-and-training)
        - [2.1.3 Evaluation and Performance Metrics](#213-evaluation-and-performance-metrics)
    - [2.2 Object Detection with YOLOv7](#22-object-detection-with-yolov7)
        - [2.2.1 Image File Reading and Dataset Organization](#221-image-file-reading-and-dataset-organization)
        - [2.2.2 Class Visualization](#222-class-visualization)
        - [2.2.3 Reformatting Data for YOLOv7](#223-reformatting-data-for-yolov7)
        - [2.2.4 Encoding Classes](#224-encoding-classes)
        - [2.2.5 Resizing Images and Annotations](#225-resizing-images-and-annotations)
        - [2.2.6 Visualizing Images and Annotations](#226-visualizing-images-and-annotations)
        - [2.2.7 Normalizing Annotations](#227-normalizing-annotations)
        - [2.2.8 Saving Images and Annotations](#228-saving-images-and-annotations)
        - [2.2.9 Installing Dependencies for YOLOv7](#229-installing-dependencies-for-yolov7)
        - [2.2.10 Training YOLOv7](#2210-training-yolov7)
        - [2.2.11 Monitoring Training Progress with WandB](#2211-monitoring-training-progress-with-wandb)
    - [2.3 Few-Shot Learning with Siamese Networks](#23-few-shot-learning-with-siamese-networks)
        - [2.3.1 Cloning Darknet Repository](#231-cloning-darknet-repository)
        - [2.3.2 Compiling Darknet](#232-compiling-darknet)
        - [2.3.3 Training Siamese Networks](#233-training-siamese-networks)
- [3. License](#3-license)
- [4. Acknowledgements](#4-acknowledgements)

## 1. Introduction

This project aims to perform multi-task learning, including image classification, object detection, and few-shot learning. It leverages CNNs for image classification, YOLOv7 for object detection, and Siamese networks for few-shot learning tasks.

## 2. Tasks

### 2.1 Image Classification with CNNs

#### 2.1.1 Data Preparation and Preprocessing

The code prepares and preprocesses the dataset for image classification tasks.

#### 2.1.2 Model Architecture and Training

It defines the architecture of the CNN model and trains it on the prepared dataset.

#### 2.1.3 Evaluation and Performance Metrics

The trained CNN model is evaluated, and performance metrics are calculated to assess its accuracy.

### 2.2 Object Detection with YOLOv7

#### 2.2.1 Image File Reading and Dataset Organization

The code reads image files and organizes the training dataset into a suitable format for YOLOv7 training.

#### 2.2.2 Class Visualization

It visualizes the distribution of classes in the dataset using bar plots.

#### 2.2.3 Reformatting Data for YOLOv7

The data is reformatted to meet the input requirements of the YOLOv7 model, including normalization of bounding box coordinates.

#### 2.2.4 Encoding Classes

The classes are encoded into numerical labels for training purposes.

#### 2.2.5 Resizing Images and Annotations

Both images and annotations are resized to a uniform size to facilitate training.

#### 2.2.6 Visualizing Images and Annotations

Random images with bounding box annotations are visualized to ensure data correctness.

#### 2.2.7 Normalizing Annotations

Annotations are normalized to fit within the range [0, 1] to match the model input.

#### 2.2.8 Saving Images and Annotations

Images and annotations are saved in directories for training and validation purposes.

#### 2.2.9 Installing Dependencies for YOLOv7

Dependencies required for YOLOv7 are installed to set up the training environment.

#### 2.2.10 Training YOLOv7

The YOLOv7 model is trained on the dataset with specified configurations.

#### 2.2.11 Monitoring Training Progress with WandB

Training progress is monitored using WandB, and evaluation metrics are visualized for analysis.

### 2.3 Few-Shot Learning with Siamese Networks

#### 2.3.1 Cloning Darknet Repository

The Darknet repository, containing the necessary code for training Siamese networks, is cloned from the official source.

#### 2.3.2 Compiling Darknet

Darknet is compiled to prepare for training the Siamese networks.

#### 2.3.3 Training Siamese Networks

Siamese networks are trained using the Darknet framework to perform few-shot learning tasks.

## 3. License

[Include license information here]

## 4. Acknowledgements

[Include acknowledgements here]
