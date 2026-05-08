# 🧠 Alzheimer's Drug Classifier: ML Virtual Screening Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)](https://www.rdkit.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

This repository contains a complete Machine Learning and Cheminformatics pipeline designed for the **virtual screening of bioactive compounds** against Alzheimer's disease. 

The model processes raw bioactivity data (IC50), calculates molecular descriptors (Morgan Fingerprints), and trains multiple algorithms to predict whether a new molecule has potential as an active drug.

## 🔬 Project Methodology

The main script (`virtual_screening_pipeline.py`) executes a 5-phase workflow:

1. **Data Curation (Data Wrangling):** 
   - Filtering the ChEMBL database.
   - Retaining exclusively standardized IC50 assays in nanomolar (nM) units.
   - Cleaning ambiguous data and handling null values.

2. **Classification and Class Balancing:** 
   - **Active (Class 1):** IC50 <= 1000 nM.
   - **Inactive (Class 0):** IC50 >= 10000 nM.
   - Elimination of the intermediate "grey zone" and application of *Undersampling* to prevent model bias.

3. **Feature Engineering:**
   - Transformation of chemical representations (SMILES) into numerical vectors using 1024-bit **Morgan Fingerprints** (radius 2) via the RDKit library.
   - Feature Selection optimization using `SelectKBest` (ANOVA F-value) to find the "Sweet Spot" of features that prevent overfitting.

4. **Model Benchmarking:**
   - Evaluation using Stratified 5-Fold Cross-Validation across 4 main algorithms:
     * Random Forest
     * Support Vector Machines (SVM)
     * XGBoost
     * K-Nearest Neighbors (KNN)
   - Visual generation of Confusion Matrices and ROC Curves.

5. **Virtual Screening and External Validation:**
   - Retraining the winning model with the optimal features.
   - Prediction of blind candidates (positive controls, negative controls, and literature findings).
   - Strict Data Leakage control to ensure validation molecules were not part of the training dataset.

## 🛠️ Requirements & Installation

To run this code in your local environment, make sure you have Python installed and run the following command to install the dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn rdkit
