# Salary Predictor using Linear Regression (Gradient Descent)

A machine learning project that predicts employee salaries based on years of experience using Linear Regression implemented from scratch with Gradient Descent.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Folder Structure](#folder-structure)
- [Implementation Details](#implementation-details)
- [Prediction Example](#prediction-example)
- [Visualization](#visualization)
- [License](#license)

## Project Overview

This project implements a Linear Regression model from scratch using Gradient Descent to predict an employee's salary based on years of experience. It demonstrates the fundamentals of machine learning model training, error calculation, and predictions without relying on high-level libraries for training.

## Dataset

The dataset used is `Salary_dataset.csv`, which contains two columns:

1. **YearsExperience** – Number of years of experience of an employee
2. **Salary** – Corresponding salary of the employee

### Sample Data

| YearsExperience | Salary |
|----------------|--------|
| 1.1            | 39343  |
| 1.3            | 46205  |
| 1.5            | 37731  |
| 2.0            | 43525  |
| 2.2            | 39891  |

## Features

- ✅ Train Linear Regression using Gradient Descent from scratch
- ✅ Predict salary for any given years of experience
- ✅ Visualize results with scatter plots and regression line
- ✅ Calculate Mean Squared Error (MSE) for model evaluation
- ✅ No dependency on high-level ML libraries for training

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KaunAnkit/linear-regression-gradient-descent.git
cd linear-regression-gradient-descent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy` - For numerical computations
- `pandas` - For data manipulation
- `matplotlib` - For data visualization
- `scikit-learn` - For data preprocessing

## How to Run

Run the main script:

```bash
python src/linear_regression.py
```

### What happens when you run:

1. ✅ Loads and preprocesses the dataset
2. ✅ Trains the model using gradient descent
3. ✅ Plots the regression line over actual data
4. ✅ Prints the Mean Squared Error (MSE)
5. ✅ Predicts salary for sample years of experience

## Folder Structure

```
Salary-Predictor/
│
├── data/
│   └── Salary_dataset.csv       # Dataset containing experience and salary data
│
├── src/
│   └── linear_regression.py     # Main Python script with model implementation
│
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files/folders to ignore in git
```

## Implementation Details

### Gradient Descent Formula

The model parameters are updated using the following formulas:

**Weight Update:**
```
w := w - α * (2/n) * Σ[xi * (yi - (w*xi + b))]
```

**Bias Update:**
```
b := b - α * (2/n) * Σ[(yi - (w*xi + b))]
```

Where:
- `w` – weight (slope of the regression line)
- `b` – bias (y-intercept)
- `α` – learning rate (step size for gradient descent)
- `n` – number of training samples
- `xi` – input feature (years of experience)
- `yi` – actual output (salary)

### Mean Squared Error (MSE)

The model's performance is evaluated using MSE:

```
MSE = (1/n) * Σ(yi - ŷi)²
```

Where `ŷi` is the predicted value.

## Prediction Example

After training, you can predict salary for any number of years of experience:

```python
from src.linear_regression import predictor

years = 5
salary = predictor(years, w, b)
print(f"Predicted Salary for {years} years of experience: ${salary:,.2f}")
```

### Sample Output

```
Years of Experience: 5
Predicted Salary: $72,000
```

## Visualization

The project generates a scatter plot showing:

```python
import matplotlib.pyplot as plt

plt.scatter(x, y, label='Actual Salaries')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.show()
```

## License

This project is open-source and available under the [MIT License](LICENSE). Feel free to modify and distribute it.

---

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## Author

**Ankit Jha** - [KaunAnkit](https://github.com/KaunAnkit)

---

⭐ If you found this project helpful, please consider giving it a star on GitHub!
