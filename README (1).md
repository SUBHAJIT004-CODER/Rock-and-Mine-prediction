
# Rock vs Mine Prediction System

## Project Overview

This project uses Machine Learning to classify underwater
objects as Rock or Mine based on sonar signal measurements.

## Algorithm

Logistic Regression (Scikit-learn).

## Dataset Information

- Total records: 208
- Input features: 60
- Rock class: R
- Mine class: M

## Model Performance

The model uses a 90:10 stratified train-test split.

The recorded test accuracy is generated during training
and saved in metadata.json.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Repository Files

- app.py: Streamlit application
- model.pkl: Trained machine learning model
- requirements.txt: Python dependencies
- metadata.json: Model information
- README.md: Project documentation

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Deployment

1. Upload the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select app.py as the main file.
6. Click Deploy.

## Disclaimer

This is an educational machine learning project.
Predictions should not be used as the sole basis for
underwater navigation or mine detection safety decisions.
