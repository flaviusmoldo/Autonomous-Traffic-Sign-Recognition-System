###### \# Autonomous Traffic Sign Recognition — GTSRB

###### 

###### Deep learning pipeline for traffic sign classification on the \[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/), a 43-class dataset with \~50,000 real-world images. Two architectures are implemented and compared: a custom CNN trained from scratch and a partially fine-tuned ResNet18.

###### 

###### \## Results

###### 

###### |   Model    | Test Accuracy | Macro F1 | Parameters | Epochs |

###### |------------|---------------|----------|------------|--------|

###### | Custom CNN |     81.3%     |   0.73   |    427K    |   20   |

###### |  ResNet18  |     94.2%     |   0.92   |    8.4M    |    8   |

###### 

###### \## Architecture Overview

###### 

###### Custom CNN — 4 convolutional blocks (Conv → BN → ReLU → MaxPool), channels 3→32→64→128→256, AdaptiveAvgPool, classifier head with Dropout(0.4). Adam + weight decay 1e-3 + CosineAnnealingLR.

###### 

###### ResNet18 — Pretrained on ImageNet. Layers 1–3 frozen, layer4 unfrozen (lr=1e-4), classification head replaced with Dropout(0.3) + Linear(512, 43) (lr=1e-3). CosineAnnealingLR over 8 epochs.

###### 

###### \## Dataset Setup

###### 

###### GTSRB images are downloaded automatically from the official ERDA mirror. The pipeline converts `.ppm` files to JPEG and splits them 90/10 into train/val per class, with the official test partition (12,630 images) held out for final evaluation only.

###### 

###### All images are resized to 224×224 and normalized with ImageNet statistics.

###### 

###### \## Requirements

###### torch

###### torchvision

###### numpy

###### matplotlib

###### scikit-learn

###### Pillow

###### 

###### Install with:

###### pip install -r requirements.txt

###### 

###### \## Usage

###### 

###### Open and run `AutonomousTrafficSignRecognitionSystem.ipynb` end-to-end. The notebook covers:

###### 

###### 1\. Dataset download and preprocessing

###### 2\. Augmentation pipelines (separate for CNN and ResNet18)

###### 3\. Training with learning rate scheduling

###### 4\. Evaluation — accuracy, per-class precision/recall/F1, confusion matrix

###### 5\. Perception demo — test images with predicted label, confidence score, and color-coded bounding box overlay

###### 

###### \## Key Design Decisions

###### 

###### \- Horizontal flipping excluded — mirroring changes the meaning of directional signs

###### \- Heavier augmentation for the CNN, lighter for ResNet18 to preserve ImageNet features

###### \- Only layer4 unfrozen in ResNet18 — full backbone freeze gave \~63% val accuracy vs \~90% with layer4 unfrozen

###### \- Seed-based reproducibility throughout

###### 

###### \## Author

###### 

###### Moldovan Flavius Cosmin — Technical University of Cluj-Napoca, Deep Learning Project

