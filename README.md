
# Salary Prediction using Simple Linear Regression

## Project Overview
This project implements Simple Linear Regression to predict an employee's salary based on their years of work experience.

It demonstrates the complete machine learning workflow, from data preprocessing and model training to prediction and visualization.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
The project uses Salary_Data.csv, which contains:
- YearsExperience: Years of professional experience.
- Salary: Corresponding employee salary.

## Project Workflow
1. Importing required libraries.
2. Loading the dataset.
3. Separating independent and dependent variables.
4. Splitting the dataset into training and testing sets (80:20).
5. Training the Simple Linear Regression model.
6. Predicting test set results.
7. Evaluating model performance.
8. Visualizing training and testing results.
9. Predicting salary for new input data.

## Model Evaluation
The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score (R2)

## How to Run

### 1. Clone the repository
git clone https://github.com/aryapatel-ibps/Salary-Prediction-using-Linear-Regression.git

### 2. Navigate to the project folder
cd Salary-Prediction-using-Linear-Regression

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the project
python main.py

## Example Prediction
The model predicts the salary of an employee with 9.3 years of experience.

## Author
Arya Patel
