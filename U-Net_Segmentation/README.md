## Project Code

#### Folder structure:
```commandline
.
├── dataset_phase_1/
│   └── segmentation_dataset/
│       ├── seg_train
│       └── seg_val
├── loaded_model_predictions
├── models
├── visualizations
├── baseline_model.py
├── dataset.py
├── Dice loss.ipynb
├── Bceloss.ipynb
├── eda.ipynb
└── readme.md
```

#### Data Location:
Folder containing data files should be in the current working directory as shown in under "dataset_phase_1" in above tree structure. 

#### model generated Files
All the model generated files are in 3 directories. All of them are generated automatically through code.
- "loaded_model_predictions" directory contains prediction generated from each type of loss under their directory name
- "models" directory contains models generated from training model using different type of loss
- "visualizations" directory contains sample prediction and plots of loss curves


#### Code Files
Base model code and dataset code is available in a separate `.py` file name. Each file is then accessed through Jupyter notebook. The details are mentioned below:

- **eda.ipynb**
  - Eda and data stats of the dataset.

- **Dice loss.ipynb**
  - Code for training and generating predictions using dice loss

- **Bceloss.ipynb**
  - Code for training and generating predictions using BCE loss

- **baseline_model.py**
  - Contains code for unet architecture

- **dataset.py**
  - Contains code for dataset class to access data


- All the paths used have been mentioned in each of the notebooks. 