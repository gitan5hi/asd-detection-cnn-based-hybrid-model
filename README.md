# MMASD Multimodal Autism Detection Project

## Overview

This project focuses on building a **multimodal deep learning pipeline** to analyze behavioral signals from children and assist in **Autism Spectrum Disorder (ASD) assessment**. The dataset used is the **MMASD (Multimodal Multiview Autism Spectrum Disorder) dataset**, which contains several behavioral modalities derived from recorded child–clinician interaction videos.

The project aims to combine multiple sources of information including:

* Optical flow from video frames (motion cues)
* 2D body skeleton keypoints (pose information)
* 3D body skeleton coordinates (depth-aware posture information)
* Clinical metadata and ADOS-2 evaluation scores

The final goal is to build a **Hybrid CNN + BiLSTM + Attention model** capable of learning temporal and multimodal behavioral patterns associated with ASD.

---

# Project Status

Current progress includes:

* Dataset preprocessing and downsampling
* Multimodal dataset loader implementation
* Optical Flow feature extraction using CNN
* Skeleton motion modeling using BiLSTM

Model fusion and training will be implemented in the next stage.

---

# Dataset Description

The project uses the **MMASD dataset**, which contains several modalities extracted from video recordings of children undergoing autism assessments.

The dataset includes:

### 1. Optical Flow

Derived from raw RGB video frames. Optical flow captures motion between consecutive frames.

Each frame has two components:

* `*_x.jpg` – horizontal motion
* `*_y.jpg` – vertical motion

Originally stored as `.npy` files, these were converted to grayscale `.jpg` images due to storage limitations.

---

### 2. 2D Skeleton Data

Generated using **OpenPose**.

Contains:

* 25 body keypoints
* Stored as `.json` files
* Each frame contains (x, y, confidence) values

---

### 3. 3D Skeleton Data

Generated using **ROMP (Regression of Multiple 3D People)**.

Contains:

* 24 SMPL body joints (used in this project)
* Stored as `.npz` files
* Each frame contains (x, y, z) coordinates

---

### 4. Clinical Assessment Data

The dataset also contains **clinician evaluation data** including:

* Subject ID
* Gender
* Chronological Age
* ADOS-2 Classification
* Autism Severity Scores

These are stored in:

```
ADOS_rating.csv
```

---

# Frame Rate Processing

The original dataset videos are recorded at:

```
30 FPS
```

To reduce computational cost and standardize sequences, the data was **downsampled to 10 FPS**.

This means:

```
1 frame kept every 3 frames
```

Downsampling was applied consistently to:

* Optical flow frames
* OpenPose 2D skeleton files
* ROMP 2D coordinate files
* ROMP 3D coordinate files

---

# Project Folder Structure

```
MMASD_project/
│
├── ADOS_rating.csv
│
├── optical_flow/
├── optimal_flow_downsampled/
│
├── 2D_skeleton/
├── 2D_openpose_downsampled/
│
├── 3D_skeleton/
├── 3d_romp_downsampled/
│
├── dataset_loader.py
├── downsampling_jpg.py
├── downsampling_json.py
├── downsampling_npz.py
│
├── cnn_optical_flow.py
├── skeleton_bilstm.py
│
├── test_loader.py
├── test_cnn.py
```

Note: The dataset folders are not included in this repository due to their large size.

---

# Data Preprocessing

Three preprocessing scripts were implemented to downsample the dataset.

### Optical Flow Downsampling

Script:

```
downsampling_jpg.py
```

Function:

* Processes image frames
* Selects every 3rd frame
* Maintains folder structure
* Saves into `optimal_flow_downsampled`

---

### OpenPose JSON Downsampling

Script:

```
downsampling_json.py
```

Function:

* Reads JSON pose files
* Keeps every third frame
* Saves processed files into `2D_openpose_downsampled`

---

### ROMP NPZ Downsampling

Script:

```
downsampling_npz.py
```

Function:

* Processes `.npz` skeleton coordinate files
* Downsamples frames
* Saves results into `3d_romp_downsampled`

---

# Dataset Loader

A custom PyTorch dataset loader was implemented to load multimodal data.

File:

```
dataset_loader.py
```

The loader reads and synchronizes:

* Optical flow frames
* 2D skeleton data
* 3D skeleton data
* Clinical metadata

Each sample returns:

```
{
    optical_flow : Tensor [T, 2, 224, 224]
    skeleton2d   : Tensor [T, 25, 3]
    skeleton3d   : Tensor [T, 24, 3]
    meta         : Tensor [age, gender]
    label        : ASD diagnosis
}
```

---

# Data Loader Output

Example batch output:

```
Optical Flow shape: torch.Size([2, 30, 2, 224, 224])
2D Skeleton shape: torch.Size([2, 30, 25, 3])
3D Skeleton shape: torch.Size([2, 30, 24, 3])
Meta shape: torch.Size([2, 2])
Label shape: torch.Size([2])
```

Meaning:

* Batch size = 2
* 30 frames per sample
* Optical flow with X and Y channels
* 2D skeleton with 25 joints
* 3D skeleton with 24 joints
* Metadata includes age and gender

---

# Optical Flow CNN Feature Extractor

File:

```
cnn_optical_flow.py
```

A modified **ResNet-18 CNN** is used to extract motion features from optical flow images.

Input:

```
(Batch, Frames, 2, 224, 224)
```

Output:

```
(Batch, Frames, 256)
```

This converts raw motion images into compact motion representations.

---

# Skeleton Motion Modeling (BiLSTM)

File:

```
skeleton_bilstm.py
```

Bidirectional LSTM networks are used to model body movement patterns over time.

Two models are used:

* 2D Skeleton BiLSTM
* 3D Skeleton BiLSTM

Input:

```
2D Skeleton: (B, 30, 25, 3)
3D Skeleton: (B, 30, 24, 3)
```

Output:

```
(B, 30, 256)
```

The model learns temporal motion patterns from body posture changes.

---

# Current Model Outputs

At this stage, three feature streams are generated:

```
Optical Flow CNN  → (B, 30, 256)
2D Skeleton BiLSTM → (B, 30, 256)
3D Skeleton BiLSTM → (B, 30, 256)
```

These features will later be fused for multimodal classification.

---

# Upcoming Work

Next development steps include:

* Multimodal feature fusion
* Attention mechanism for temporal importance
* Final classification network
* Model training and evaluation
* Performance comparison across modalities

---

# Requirements

Main dependencies:

```
Python 3.9+
PyTorch
Torchvision
NumPy
Pandas
OpenCV
Pillow
```

Install using:

```
pip install torch torchvision numpy pandas opencv-python pillow
```

---

# Notes

* The MMASD dataset is not included in this repository due to size restrictions.
* Folder paths must be updated to match the dataset location on the local machine.

---

# Author

Project developed as part of research work on **multimodal deep learning for autism behavior analysis**.
