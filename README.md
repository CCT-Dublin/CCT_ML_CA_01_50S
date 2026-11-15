## 📌 Project Overview

This project investigates the application of Machine Learning classification algorithms to the Breast Cancer Wisconsin dataset. The main objective is to predict whether a breast tumor is **Benign** or **Malignant** based on numerical diagnostic features.

Two classification algorithms, Random Forest and K-Nearest Neighbors (KNN), are implemented and compared to evaluate their predictive performance.

The project also investigates the importance of feature selection, data preprocessing, cross-validation, and hyperparameter optimization. Particular attention is given to Malignant Recall, as correctly identifying malignant cases is an important consideration in this classification problem.

## 🧪 Methodology

The project follows a structured Machine Learning workflow, covering the following stages:

1. **Data Loading and Understanding:** Inspecting the dataset structure, dimensions, data types, and data quality.
2. **Exploratory Data Analysis (EDA):** Investigating descriptive statistics, target distribution, numerical feature distributions, and correlations.
3. **Data Cleaning and Transformation:** Removing irrelevant features, preparing the target variable, applying feature scaling where required, and assessing class distribution.
4. **Feature Selection:** Comparing correlation-based selection, SelectKBest using ANOVA F-test, and Random Forest feature importance.
5. **Training and Test Splitting:** Function that will be used throughout the entire project
6. **Predictive Modeling:** Developing Random Forest and K-Nearest Neighbors classification models.
7. **Cross-Validation:** Evaluating model performance using Stratified 5-Fold Cross-Validation across three test-size configurations: 5%, 15%, and 25%.
8. **Hyperparameter Tuning:** Applying GridSearchCV using a 15% test split to identify suitable hyperparameter configurations for both models.
9. **Model Evaluation and Comparison:** Comparing Accuracy, Cross-Validation results, Malignant Recall, and Confusion Matrices, as well as investigating overfitting, underfitting, and generalization.

### Classification Experiments

Two classification algorithms are investigated:

| Model | Description |
|---|---|
| **Random Forest Classifier** | An ensemble learning algorithm that combines multiple decision trees to produce classification predictions. |
| **K-Nearest Neighbors (KNN)** | A distance-based classification algorithm that predicts a class based on the nearest observations in the feature space. |

Both models are evaluated using the same three test-size configurations:

- **5% Test / 95% Training**
- **15% Test / 85% Training**
- **25% Test / 75% Training**

Stratified 5-Fold Cross-Validation is used to assess model performance and consistency. GridSearchCV is subsequently applied to optimize the hyperparameters of each classifier.

The evaluation considers the following metrics:

- Accuracy
- Cross-Validation Mean and Standard Deviation
- Malignant Precision
- Malignant Recall
- Malignant F1-score

Confusion matrices and comparative visualizations are also used to investigate classification errors and support the final model selection.

## 📁 Project Structure

```text
CCT_ML_CA_01_50S/
│
├── data/
│   └── data.csv
│
├── docs/
│   ├── CA1 - Machine Learning (Sept 2026) - FT.docx
│   ├── CCT Assessment Cover Sheet (Approved August 2026).docx
│   ├── CCT_ML_CA_01_50S_2026438.docx
│   └── Harvard_Referencing_Handbook__includes_AI_new.pdf
│
├── notebooks/
│   └── CCT_ML_CA_01_50S_2026438.ipynb
│
├── src/
│   ├── __init__.py
│   ├── barplot.py
│   ├── boxplot.py
│   ├── helpful.py
│   └── histplot.py
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

## ⚙️ Environment Setup

It is recommended to create a dedicated Python virtual environment before installing the project dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/CCT-Dublin/CCT_ML_CA_01_50S.git
cd CCT_ML_CA_01_50S
```

### 2. Create a virtual environment

**macOS / Linux / Windows**
```bash
conda create -n $(basename $(pwd)) python=3.14.10
```

### 3. Activate the virtual environment

**macOS / Linux**
```bash
conda activate $(basename $(pwd))
```

### 4. Install dependencies

Install the project dependencies:

```bash
pip install -r requirements-dev.txt
```


> Note: If `requirements-dev.txt` already includes `requirements.txt`, installing the development requirements may be sufficient. Otherwise, install both files as shown above.

## ▶️ Running the Project

### 1. Start JupyterLab

From the project root directory, execute:

```bash
jupyter lab
```

### 2. Open the notebook

Navigate to:

```text
notebooks/CCT_ML_CA_01_50S_2026438.ipynb
```

### 3. Select the virtual environment kernel

Make sure the notebook is using the Python interpreter from the `.venv` environment.


### 4. Execute the notebook

Run the notebook cells sequentially, starting from the first cell.

**Important:** The notebook uses relative paths to access the dataset and project modules. Open and execute it with `notebooks/` as the working directory so that paths such as `../data/` and `../src/` resolve correctly.


## 🛠️ Technologies and Libraries


## 📚 Documentation and References

References used throughout the project follow the Harvard referencing style.


## ⚠️ Academic Integrity and AI Declaration

This is an individual academic assignment.

All technical implementations, data analysis, model evaluation, and interpretation of results are undertaken and verified by the student.

Generative AI tools, where permitted by the assignment guidelines, are used only as supporting tools for activities such as brainstorming, planning, research assistance, and improving written communication.

All permitted AI usage will be documented in accordance with the CCT College Dublin AI Declaration requirements. Supporting academic documentation, assignment guidelines and the [AI Declaration](./docs/CCT_ML_CA_01_50S_2026438.docx) are available in the `docs/` directory.

## 🎓 Academic Information

| Field | Details |
|---|---|
| Institution | CCT College Dublin |
| Programme | Higher Diploma in Science in Data Analytics for Business |
| Module | HDSDAB Sep26 FT | Machine Learning |
| Assignment | CA1 – Machine Learning |
| Assignment Type | Individual |
| Weighting | 50% |
| Submission Date | 15 November 2026 |

---

## Document Version

| Versão Do Document |       Author     |      Editor      |    Data    |  Percentage Complete  |
|        :---:       |       :---:      |       :---:      |    :---:   |         :---:         |
|        1.0.0       | Matheus S. Silva | Matheus S. Silva | 2026-11-15 |          95%          |