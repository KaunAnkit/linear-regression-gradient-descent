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
- ✅ Random prediction generator (1-20 years experience)
- ✅ No dependency on high-level ML libraries for training

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KaunAnkit/salary-predictor.git
cd salary-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy` - For numerical computations
- `pandas` - For data manipulation
- `matplotlib` - For data visualization

## How to Run

Run the main script:

```bash
python linear_regression.py
```

### What happens when you run:

1. ✅ Loads and preprocesses the dataset
2. ✅ Trains the model using gradient descent (1400 epochs)
3. ✅ Prints trained parameters (w, b) and MSE
4. ✅ Plots the regression line over actual data
5. ✅ Predicts salary for a random year of experience (1-20 years)

## Folder Structure

```
Salary-Predictor/
│
├── data/
│   └── Salary_dataset.csv       # Dataset containing experience and salary data
│
├── linear_regression.py         # Main Python script with model implementation
│
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files/folders to ignore in git
```

## Implementation Details

### Model Parameters

```python
w = 0                    # Initial weight (slope)
b = 0                    # Initial bias (intercept)
learning_rate = 0.01     # Learning rate
epochs = 1400            # Number of training iterations
```

### Gradient Descent Algorithm

The model parameters are updated using gradient descent:

**Step 1: Make predictions**
```python
y_pred = w * x + b
```

**Step 2: Calculate error**
```python
error = y - y_pred
```

**Step 3: Calculate gradients**
```python
dw = (-2/n) * np.sum(x * error)
db = (-2/n) * np.sum(error)
```

**Step 4: Update parameters**
```python
w -= learning_rate * dw
b -= learning_rate * db
```

Where:
- `w` – weight (slope of the regression line)
- `b` – bias (y-intercept)
- `learning_rate` – step size for gradient descent (0.01)
- `n` – number of training samples
- `x` – input feature (years of experience)
- `y` – actual output (salary)

### Mean Squared Error (MSE)

The model's performance is evaluated using MSE:

```python
MSE = np.mean((y - y_pred)**2)
```

### Training Loop

```python
for _ in range(epochs):
    y_pred = w * x + b
    error = y - y_pred
    dw = (-2/n) * np.sum(x * error)
    db = (-2/n) * np.sum(error)
    w -= learning_rate * dw
    b -= learning_rate * db
```

## Prediction Example

After training, you can predict salary for any number of years of experience:

```python
def predictor(years, w, b):
    return w * years + b

# Example prediction
years = 5
salary = predictor(years, w, b)
print(f"Predicted Salary: {salary}")
```

The script also generates a random prediction:

```python
import random

k = random.randint(1, 20)
print("Years of Experience:", k)
print("Predicted Salary:", predictor(k, w, b))
```

### Sample Output

```
   YearsExperience  Salary
0              1.1   39343
1              1.3   46205
2              1.5   37731
3              2.0   43525
4              2.2   39891

Trained w: 9449.962321455077 b: 25792.20019866871 MSE: 31635570.303868428

Years of Experience: 7 
Predicted Salary: 91941.9562502185
```

## Visualization

The project generates a scatter plot showing:
```python
plt.scatter(x, y)
plt.plot(x, y_final_pred, color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
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
