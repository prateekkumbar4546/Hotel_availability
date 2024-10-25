import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.compose import make_column_selector

# Load the data
train_data = pd.read_csv('train.csv')

# EDA: Checking null values
print("Null values:\n", train_data.isnull().sum())
print("\nData distribution:\n", train_data.describe())

# Visualize pair plots and correlation matrix
sns.pairplot(train_data)
plt.show()

correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Feature Engineering
def create_additional_features(df):
    df['price_per_night'] = df['cost'] / df['minimum_nights']
    df['review_frequency'] = df['reviews_per_month'] / df['number_of_reviews']
    df['is_entire_home'] = df['accommodation_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
    df['log_cost'] = np.log1p(df['cost'])
    df['review_to_cost_ratio'] = df['number_of_reviews'] / df['cost']
    df['high_review_count'] = df['number_of_reviews'].apply(lambda x: 1 if x > 50 else 0)
    return df

train_data = create_additional_features(train_data)

# Separate features and target variable
X = train_data.drop(columns=['yearly_availability'])
y = train_data['yearly_availability']

# Identify numerical and categorical features automatically
numeric_features = make_column_selector(dtype_include=np.number)(X)
categorical_features = make_column_selector(dtype_exclude=np.number)(X)

# Preprocess data
X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
X[numeric_features] = StandardScaler().fit_transform(X[numeric_features])

X[categorical_features] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_features])
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"Classification Report for {name}:\n", classification_report(y_val, y_pred), "\n")

# (Optional) Test set and submission file creation would follow similarly based on your test data.
