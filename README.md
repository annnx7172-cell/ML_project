# Student Performance Prediction - Ananya Singh

## Introduction About the Data

**The dataset**: The goal is to predict a student's `math_score` based on other performance indicators and demographic features (Regression Analysis).

There are 7 independent variables:

- `gender` : Gender of the student (male/female)
- `race_ethnicity` : Ethnic group of the student (group A to E)
- `parental_level_of_education` : The highest level of education completed by the student's parent(s)
- `lunch` : Type of lunch the student receives (standard / free or reduced)
- `test_preparation_course` : Whether the student completed a test preparation course (completed / none)
- `reading_score` : Score obtained in reading (out of 100)
- `writing_score` : Score obtained in writing (out of 100)

Target variable:

- `math_score` : Score obtained in mathematics (out of 100)

Dataset Source Link : [Students Performance in Exams - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Live Deployment Link

Render Deployment : [Predict Your Maths Score](http://ml-project-predict-your-maths-score.onrender.com/)

> Note: the app is hosted on Render's free tier, so it may take 30-50 seconds to "wake up" if it hasn't been visited recently.

## Approach for the Project

1. **Data Ingestion** :
   - In the Data Ingestion phase, the raw data is first read as a CSV file.
   - The data is then split into training and testing sets (80/20 split) and saved as separate CSV files inside the `artifacts/` folder.

2. **Data Transformation** :
   - A `ColumnTransformer` pipeline is built to handle numerical and categorical features differently.
   - For **numerical variables** (`reading_score`, `writing_score`): missing values are imputed using the **median**, followed by **Standard Scaling**.
   - For **categorical variables**: missing values are imputed using the **most frequent** value, followed by **One-Hot Encoding**, then scaled (without centering, to preserve sparsity).
   - This fitted preprocessor is saved as a pickle file (`preprocessor.pkl`) for reuse during prediction.

3. **Model Training** :
   - Multiple regression algorithms are trained and compared: Random Forest, Decision Tree, Gradient Boosting, Linear Regression, Ridge Regression, Lasso Regression, XGBoost, CatBoost, AdaBoost, K-Neighbors Regressor, and a Voting Regressor ensemble.
   - Each model is hyperparameter-tuned using `GridSearchCV` with cross-validation.
   - Model performance is compared using the **R² score** on the held-out test set.
   - The best performing model — **Lasso Regression** (R² ≈ 0.88) — is selected and saved as `model.pkl`.

4. **Prediction Pipeline** :
   - Converts user input into a pandas DataFrame matching the training schema.
   - Loads the saved preprocessor and model from disk, applies the same transformations used during training, and returns the predicted math score.

5. **Flask App Creation** :
   - A Flask web application provides a simple user interface where a user can input student details and receive a predicted math score in real time.
   - Deployed on **Render** (free tier).

## Model Comparison (R² Scores)

| Model | R² Score |
|---|---|
| Linear Regression | 0.8804 |
| **Lasso Regression** | **0.8806** |
| Gradient Boosting | 0.8756 |
| Voting Regressor | 0.8631 |
| CatBoosting Regressor | 0.8614 |
| AdaBoost Regressor | 0.8541 |
| Random Forest | 0.8497 |
| XGBRegressor | 0.8492 |
| Decision Tree | 0.7480 |
| K-Neighbors Regressor | 0.5447 |

## Tech Stack

- **Language**: Python 3.11
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, pandas, NumPy
- **Web Framework**: Flask
- **Deployment**: Render
- **Version Control**: Git & GitHub

## Project Structure

```
ML_project/
├── artifacts/              # Saved datasets, preprocessor, and trained model
├── notebook/
│   └── data/                # Raw dataset (stud.csv)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── app.py
├── requirements.txt
└── README.md
```

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/annnx7172-cell/ML_project.git
cd ML_project

# Create and activate a virtual environment
conda create -p venv python=3.11 -y
conda activate ./venv

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python src/components/data_ingestion.py

# Run the Flask app
python app.py
```

Then open `http://127.0.0.1:5001` in your browser.

## Exploratory Data Analysis Notebook

Link : [EDA Notebook](https://github.com/annnx7172-cell/ML_project/tree/main/notebook)

## Author

**Ananya Singh**
GitHub: [@annnx7172-cell](https://github.com/annnx7172-cell)