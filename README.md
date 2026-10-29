# Retail Demand Forecasting & Sales Prediction System

## Project Overview
A machine learning project that analyzes historical retail sales and predicts future sales using time-based features and regression models.

## Project Workflow
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training and MLflow Experiment Tracking
5. Model Evaluation and Reporting

## Dataset
This project uses the **Rossmann Store Sales** dataset from Kaggle.

Original dataset: [Rossmann Store Sales – Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

The cleaned datasets are stored in `data/cleaned/`:

- `train_cleaned.parquet` — Historical training data, including sales.
- `test_cleaned.parquet` — Data for the final forecasting period; it does not contain actual sales or customer counts.
- `store_cleaned.parquet` — Store metadata.

See `data/cleaned/notes_data.md` for data-cleaning decisions, dataset details, and instructions.

## Repository Structure
- `data/cleaned/` — Cleaned datasets and data documentation.
- `data/processed/` — Processed datasets and time-based data splits.
- `notebooks/` — Jupyter notebooks for exploration and analysis.
- `src/` — Reusable Python modules and pipelines.
- `models/` — Saved trained models.
- `predictions/` — Validation and test predictions.
- `reports/` — EDA findings, evaluation results, and figures.

## Setup
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Load the Cleaned Data
```python
import pandas as pd

train = pd.read_parquet("data/cleaned/train_cleaned.parquet")
test = pd.read_parquet("data/cleaned/test_cleaned.parquet")
store = pd.read_parquet("data/cleaned/store_cleaned.parquet")
```

## Team Responsibilities
- **Rahma** — Project Lead & Data Engineer
- **Ahmed** — EDA, Visualization & MLflow Setup
- **Mohammed** — Feature Engineering
- **Elsaeed** — ML Modeling
- **Ali** — Evaluation & Reporting

## Project Status
Data cleaning is complete, and the cleaned datasets and documentation are available in the repository. EDA, feature engineering, model training, and evaluation will be completed as the project progresses.
