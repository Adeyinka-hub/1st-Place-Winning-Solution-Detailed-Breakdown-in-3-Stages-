# **1st Place Winning Solution Detailed Breakdown ( in 3 Stages ):**

# STAGE 1: DATA PREPARATION
**Loading the Data**
The training data, stored in a CSV file, is loaded into a Pandas DataFrame. The paths to the data directories are set up to make it easy to reference them later. This step organizes the environment and ensures the necessary files are accessible.
### Preparing the Data
In this step:
The number of samples in each class is counted and stored in a dictionary. A new column is added to the DataFrame to associate each sample with the count of its class.
The paths to the images are constructed and added as a column, making it easy to reference image files.
The class names (e.g., "Tomato_Late_Blight") are converted into numerical labels using a mapping dictionary. This transformation is necessary for machine learning models, as they operate on numerical inputs rather than text labels.

**Visualizing the Data**
The frequency of samples in each class is plotted as a bar chart. This helps understand the distribution of the data and identify any imbalance among the classes. For example, if some classes have very few samples compared to others, it could signal the need for special handling, such as oversampling or data augmentation.
### Filtering the Data
**The dataset is divided based on the number of samples per class:**
Classes with fewer than 1000 samples are isolated into a separate DataFrame for special handling. These might be minority classes that need to be included in all training folds to ensure the model sees enough examples of them.
The remaining classes (those with more than 1000 samples) are kept as the primary dataset for cross-validation.
## Creating Cross-Validation Folds
To evaluate the model reliably, the data is split into 5 folds using a stratified group K-fold strategy. This ensures:
The class distribution in each fold matches the overall dataset distribution (stratification).
Samples from the same group (e.g., images with the same  Image_ID)  are kept together in the same fold, preventing data leakage. Each sample  is assigned a fold number, indicating whether it belongs to the training or validation set for a particular fold.
## Organizing the Data for Each Fold
Directories are created to organize training and validation images and labels for each fold. For each fold:
Training samples include all images not in the current fold, plus minority class samples to ensure those classes are well-represented during training.
Validation samples are taken from the current fold. The images and their corresponding YOLO-style annotation files (labels) are copied into the respective directories.
## Preparing for YOLO Training
A configuration file (data_fold_3.yaml) is generated for one of the folds. This file specifies:
The paths to the training, validation, and test datasets.
The number of classes and their names. This YAML file is required to train object detection models using YOLO, as it provides the model with the necessary dataset structure and metadata.
# STEP 2: TRAINING ROUTINE
**Train Settings Initialization**
A dictionary named params is defined to configure various training settings, including:
- Using pre-trained weights ('pretrained': True).
- Optimizer ('AdamW'), learning rate ('lr0': 3e-4), and mixed precision training ('half': True).
- Other parameters like caching, verbosity, augmentation, and save period are configured to control training behavior.
## Model Loading
The YOLO model is loaded using the ultralytics.YOLO class, specifying a pre-trained model weight file (yolo11s.pt).
## Training the Model
The model is trained on a dataset defined for a specific fold (data_fold_3.yaml).
**Key configurations include:**
- Task type ('detect') for object detection.
- Image size (1024x1024), batch size (16), and number of training epochs (50).
- Mixed precision training (amp=True) and GPU acceleration (device='cuda:0').
- Learning rate schedule, momentum, and weight decay for optimization.
- A close mosaic option (close_mosaic=30) to control data augmentation during later training stages.
- Other settings like validation (val=True), saving checkpoints (save=True), and ensuring the setup can resume interrupted training (exist_ok=True).
The training process divides the dataset into training and validation sets based on the fold, optimizing the model for detection tasks.
# STAGE 3: INFERENCE
**Dataset Paths**
## Defines paths for data, including:
- DATA_DIR: Root directory of the dataset.
- IMAGES_DIR: Directory for images.
- TRAIN_CSV, TEST_CSV: CSV files containing metadata.
- SAMPLE_SUBMISSION: Template file for submission.
## Class Mapping
- A dictionary (classes) maps crop diseases and healthy classes to integer IDs.
- Reverse mappings (e.g., id_to_class) allow for converting predictions back to human-readable labels.
## Custom Dataset Class
Implements a PyTorch Dataset class (TestRAIL) to handle test images:

**__init__:   Accepts image paths and optional transformations.**

**__getitem__:  Loads, processes, and normalizes an image.**

**__len__:  Returns the number of test samples.**

## Transformations
- Uses albumentations for image preprocessing during inference:
- Resizes images to 1024x1024.
- Normalizes pixel values using ImageNet mean and standard deviation.
- Converts the image to a PyTorch tensor using ToTensorV2.
## DataLoader
- Creates a DataLoader for test data with a batch size of 1 (likely due to varying image sizes or GPU constraints).
## YOLO Model Initialization
- Loads pre-trained YOLO models (saved from training) into a list (model_folds).
## Sets up WBF parameters:
- iou_thr: Intersection over union threshold for merging boxes.
- skip_box_thr: Minimum confidence score for a box to be considered.
- weights: Equal weights for each YOLO model in the ensemble (even though only one model was used).
## WBF Prediction Function
- Defines a function, run_wbf_on_image, that:
- Runs inference on an image using all YOLO models in the ensemble.
- Collects bounding box predictions, confidence scores, and class labels.
- Applies WBF to merge predictions from different models.
- Scales the normalized bounding boxes back to the original image dimensions.
## Inference Loop
- Iterates through the test dataset using the DataLoader.
## For each image:
- Runs WBF-based prediction.
- If no predictions are found, defaults to labeling the image as "Corn_Healthy."
- Appends predictions (class name, confidence score, bounding box coordinates) to a list (info_).
## Save Results
- Converts the prediction list (info_) into a DataFrame with columns:
- Image_ID, class, confidence, xmin, ymin, xmax, ymax.
- Saves the DataFrame as a CSV file for submission.
