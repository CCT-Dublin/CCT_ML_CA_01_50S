# 🚧 IN PROGRESING ..... 🚧

## 📌 Project Overview
## 🧪 Methodology
### Classification Experiments

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

**macOS / Linux**
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
|        0.2.0       | Matheus S. Silva | Matheus S. Silva | 2026-11-08 |          45%          |