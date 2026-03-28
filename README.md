# Customer Churn Prediction

A Streamlit-based machine learning app that predicts telecom customer churn risk from customer details and suggests retention actions.

## Overview

This project uses a trained churn prediction model and a simple Streamlit interface to estimate whether a telecom customer is likely to leave. Users can enter customer profile, service usage, contract details, and billing information, then get a churn probability score with a basic retention recommendation.

## Features

- Predicts customer churn risk from input customer details
- Shows churn probability with a visual progress bar
- Classifies customers as likely to stay or churn
- Suggests basic retention actions based on predicted risk
- Includes a training notebook and saved model artifacts

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- XGBoost

## Project Structure

```text
customer-churn-prediction/
|-- app/
|   `-- app.py
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- encoders.pkl
|-- model.pkl
|-- requirements.txt
|-- train.ipynb
`-- README.md
```

## Dataset

The project uses the Telco Customer Churn dataset stored in:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit app from the project root:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal, usually:

`http://localhost:8501`

## How It Works

1. The app loads the trained model from `model.pkl`.
2. Saved label encoders are loaded from `encoders.pkl`.
3. User inputs are transformed into the format expected by the model.
4. The app predicts churn probability.
5. A risk message and retention suggestion are displayed.

## Deployment

This project can be deployed easily on Streamlit Community Cloud.

Deployment settings:

- Repository: your GitHub repository
- Branch: `main` or `master`
- Main file path: `app/app.py`

## Future Improvements

- Add model performance metrics to the app
- Show feature importance or prediction explanations
- Add batch prediction from CSV upload
- Improve UI styling and validation

## Author

Sujal
