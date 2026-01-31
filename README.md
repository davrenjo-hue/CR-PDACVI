Overview
This project implements a radiomics consensus model for pancreatic ductal adenocarcinoma (PDAC) diagnosis and vascular involvement prediction, using the PANORAMA dataset and integrating radiomic features from 5 pancreatic-related anatomical structures (artery, vein, pancreatic parenchyma, pancreatic duct, common bile duct).
.
├── Modeling of various structures/  # Single-structure model training
│   └── modeling.py                 # Train structure-specific classifiers
├── feature/                        # Radiomic feature processing
│   ├── Sample_alignment.py         # Data alignment
│   ├── feature_extraction.py       # Radiomic feature extraction (PyRadiomics)
│   ├── feature_selection.py        # LASSO-based feature selection
│   └── features.yaml               # Feature extraction parameters
└── fusion/                         # Model fusion
    └── stacking.py                 # Stacking ensemble for consensus model
