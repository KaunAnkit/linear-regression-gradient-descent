# linear-regression-gradient-descent

Salary Predictor using Linear Regression (Gradient Descent)
Table of Contents

Project Overview

Dataset

Features

Installation

How to Run

Folder Structure

Implementation Details

Prediction Example

Visualization

License

Project Overview

This project implements a Linear Regression model from scratch using Gradient Descent to predict an employee's salary based on years of experience.
It demonstrates the basics of machine learning model training, error calculation, and predictions without relying on high-level libraries for training.

Dataset

The dataset used is Salary_dataset.csv, which contains two columns:

YearsExperience – number of years of experience of an employee

Salary – corresponding salary of the employee

Example of the dataset:

YearsExperience	Salary
1.1	39343
1.3	46205
1.5	37731
2.0	43525
2.2	39891
Features

Train Linear Regression using Gradient Descent

Predict salary for any given years of experience

Visualize results with scatter plots and regression line

Calculate Mean Squared Error (MSE)

Installation

Clone the repository:

git clone https://github.com/KaunAnkit/linear-regression-gradient-descent.git
cd linear-regression-gradient-descent


Install dependencies:

pip install -r requirements.txt


Dependencies:

numpy

pandas

matplotlib

scikit-learn

How to Run

Run the main script:

python src/linear_regression.py


This will:

Train the model using gradient descent

Plot the regression line

Print the Mean Squared Error (MSE)

Predict salary for a random year of experience

Folder Structure
Salary-Predictor/
│
├── data/
│   └── Salary_dataset.csv       # Dataset
│
├── src/
│   └── linear_regression.py     # Main Python script
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files/folders to ignore

Implementation Details

Gradient Descent Formula:

𝑤
:
=
𝑤
−
𝛼
2
𝑛
∑
𝑖
=
1
𝑛
𝑥
𝑖
(
𝑦
𝑖
−
(
𝑤
𝑥
𝑖
+
𝑏
)
)
w:=w−α
n
2
	​

i=1
∑
n
	​

x
i
	​

(y
i
	​

−(wx
i
	​

+b))
𝑏
:
=
𝑏
−
𝛼
2
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
(
𝑤
𝑥
𝑖
+
𝑏
)
)
b:=b−α
n
2
	​

i=1
∑
n
	​

(y
i
	​

−(wx
i
	​

+b))

w – weight (slope)

b – bias (intercept)

α – learning rate

n – number of samples

Mean Squared Error (MSE) is used to evaluate the model:

𝑀
𝑆
𝐸
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
𝑖
^
)
2
MSE=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−
y
i
	​

^
	​

)
2
Prediction Example

After training, you can predict a salary for any number of years of experience:

from src.linear_regression import predictor

years = 5
salary = predictor(years, w, b)
print(f"Predicted Salary for {years} years of experience: {salary}")


Sample output:

Years of Experience: 5
Predicted Salary: 72000

Visualization

The project also plots a scatter plot of actual salaries and the regression line:

Blue dots – actual salaries

Red line – predicted salaries by the model

plt.scatter(x, y)
plt.plot(x, y_final_pred, color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

License

This project is open-source and free to use. You can modify and distribute it under the MIT License
.
