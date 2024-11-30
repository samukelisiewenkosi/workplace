# Job Market Analysis - Machine Learning Project
![ai_pic](ai_pic.png)
### Project Overview
This project analyzes a large dataset of job market information to uncover key insights about the evolving dynamics across various industries. The analysis focuses on the intersection of AI adoption, automation risk, and required skills and how they impact salaries, job growth projections, and remote work trends.

The aim is to provide actionable insights for job seekers, companies, and policymakers to better understand how technological advancements are reshaping the workforce. The analysis also helps to identify the industries and roles likely to experience growth or disruption, as well as regions offering the best compensation.

Key Questions Addressed
How are AI adoption and automation risks affecting job roles and industries?
What are the trends in required skills across industries?
How do salaries vary across locations and industries?
What is the prevalence of remote work options, and how does it correlate with salary and industry?
Which jobs and industries are projected to grow?
Data Description
The dataset used in this project contains the following columns:

Job_Title: The title of the job.
Industry: The industry in which the job is located.
Company_Size: The size of the company (e.g., small, medium, large).
Location: The geographic location of the job.
AI_Adoption_Level: The level of AI adoption in the industry (e.g., high, medium, low).
Automation_Risk: The risk of the job being automated.
Required_Skills: The skills required for the job.
Salary_USD: The salary offered for the role in USD.
Remote_Friendly: Whether the job is remote-friendly.
Job_Growth_Projection: The projected growth rate of the job/industry.
Goals
The goal of this machine learning project is to build predictive models that can:

Predict salary based on the features such as job title, industry, and automation risk.
Predict job growth projections based on AI adoption levels, automation risks, and required skills.
Analyze trends in remote work and salaries to provide actionable insights.
Challenges
The key challenges include:

Handling missing data and outliers in the dataset.
Feature engineering for categorical variables such as job title, industry, and location.
Training models and evaluation
Tuning machine learning models to improve prediction accuracy.
Interpreting complex relationships between automation, AI, and workforce trends.
Installation
To set up the project on your local machine, follow the steps below.

Prerequisites
Ensure that you have the following installed:

Python 3.x
pip (Python package manager)
Install Required Libraries
Clone the repository:

git clone https://github.com/samukelisiewenkosi/workplace.git
cd workplace
Install the required Python libraries:

pip install -r requirements.txt
The requirements.txt file includes the following dependencies:

pandas: For data manipulation and analysis.
scikit-learn: For machine learning algorithms.
matplotlib and seaborn: For data visualization.
numpy: For numerical operations.
Usage
1. Load the Dataset
You can load the dataset using the following code:

python
import pandas as pd

### Load the dataset
file_path = 'ai_job_market_insights.csv'  
df = pd.read_csv(file_path)

### Check the first few rows of the data
print(df.head())
2. Data Preprocessing
Before applying machine learning models, you should preprocess the data by:

Handling missing values.
Converting categorical variables into numerical values (e.g., using one-hot encoding).
Scaling features if necessary.
3. Building Machine Learning Models
You can train models to predict salaries and job growth projections. Here's an example of using Linear Regression for salary prediction:

python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

### Preprocess data (encoding, scaling, etc.)
### Example: Drop the target column and encode categorical features
X = df.drop(columns=['Salary_USD'])
y = df['Salary_USD']

### One-hot encoding of categorical columns
X = pd.get_dummies(X, drop_first=True)

### Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

### Make predictions
y_pred = model.predict(X_test)

### Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

4. Model Evaluation
Evaluate the performance of the model using metrics such as Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE).

5. Visualizations
Use matplotlib and seaborn for visualizations such as:

Salary distribution across different industries.
Job growth projections by industry.
python
import seaborn as sns
import matplotlib.pyplot as plt

### Visualize the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
Contributing
Contributions to the project are welcome. If you'd like to contribute, please fork the repository and submit a pull request.

