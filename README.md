# Titanic Survival — Supervised Machine Learning

Supervised machine-learning classification of Titanic passenger survival using KNN, Decision Tree, and Random Forest with preprocessing, model evaluation, and feature importance.

## Overview

This repository presents a reconstructed portfolio version of a **Data Mining group project completed during WS 2019/2020** in the Master's program at Cologne University of Applied Sciences.

The original project used the Kaggle **Titanic: Machine Learning from Disaster** dataset to investigate passenger-survival patterns and compare three supervised classification algorithms:

- k-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

The original Jupyter notebook is no longer available. The notebook in this repository was therefore reconstructed from the surviving 2020 code export and the accompanying group report.

## Machine-Learning Workflow

The project demonstrates a complete classical supervised-learning workflow:

- Exploratory data analysis
- Missing-value handling
- Categorical feature encoding
- Feature selection
- Train/test splitting
- KNN classification
- Decision Tree classification
- Random Forest classification
- Confusion-matrix evaluation
- Accuracy, precision, and recall
- Decision-tree feature importance
- Decision-tree visualization
- Passenger-level prediction
- Kaggle test-set prediction export

## Dataset

The project uses the Kaggle Titanic competition data.

Expected files:

```text
data/
├── train.csv
└── test.csv
```

The raw datasets are not redistributed in this repository.

Source: Kaggle — Titanic: Machine Learning from Disaster  
https://www.kaggle.com/c/titanic

See [`data/README.md`](data/README.md) for setup instructions.

## Models

### k-Nearest Neighbors

The reconstructed notebook preserves the settings visible in the surviving code export:

- `n_neighbors=5`
- Minkowski distance
- `p=2` (Euclidean distance)

### Decision Tree

The Decision Tree uses entropy as the split criterion and serves as the main interpretable classifier in the original project.

### Random Forest

The Random Forest uses:

- 68 estimators
- entropy criterion
- fixed random state

## Historical Results

The surviving project materials contain two slightly different experiment versions.

### Results documented in the written report

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| KNN | 75% | 72% | 64% |
| Decision Tree | 80% | 79% | 71% |
| Random Forest | 81% | 82% | 68% |

### Results preserved in the code-export PDF

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| KNN | 69.1% | 62.9% | 60.3% |
| Decision Tree | 75.3% | 71.6% | 65.8% |
| Random Forest | 75.8% | 74.2% | 63.0% |

These values should not be combined as though they came from the same run. The report and code export contain different parameter and random-state choices.

The reconstructed notebook therefore computes fresh metrics when executed and preserves the historical numbers only for archival context.

## Feature Importance

The original work used Decision Tree feature importance to examine which passenger attributes most influenced the classifier.

The surviving code export ranked the strongest features as:

1. Sex
2. Age
3. Fare
4. Passenger Class

followed by `SibSp`, `Parch`, and `Embarked`.

## Repository Structure

```text
Titanic-Survival-Supervised-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   └── Titanic_Survival_Classification_Reconstructed.ipynb
└── report/
    └── README.md
```

## Running the Notebook

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then place `train.csv` and `test.csv` in `data/` and start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/Titanic_Survival_Classification_Reconstructed.ipynb
```

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Original Project and Collaboration

The original project was completed as **Group 5** during the Data Mining course in WS 2019/2020.

Original group members:

- Gerta Mata
- Nafiu Ikeoluwa Hammed
- Onassis Sowah Anyetei

The original report's workload table identifies Nafiu Ikeoluwa Hammed's contributions across theoretical work, data cleansing, train/test splitting and fitting, algorithm comparison, the real-life prediction scenario, and conclusions.

This repository does **not** claim sole authorship of the original group work.

## Reconstruction Note

The original `.ipynb` file is no longer available. This portfolio version was reconstructed from:

- the surviving PDF export of the Jupyter code and outputs
- the final written group report

The reconstruction modernizes deprecated Python/Scikit-learn syntax where needed while preserving the historical analytical workflow and clearly separating reproduced outputs from historical reported results.

## Privacy Note

The original report PDF is not included in this repository because its cover contains student matriculation numbers for all three group members.

## Portfolio Note

The original academic work was completed in **2020**. This repository was later curated and reconstructed for portfolio presentation.
