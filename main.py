import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Load the data with features
train_data = pd.read_csv('train.csv')

# EDA: Checking null values
print(train_data.isnull().sum())

# Reason: It's crucial to identify and handle missing values to prevent issues during model training.

# EDA: Distribution of columns
print(train_data.describe())

# Reason: Understanding the distribution of numerical features helps in identifying outliers and the need for scaling.

# EDA: Visualize pair plots
sns.pairplot(train_data)
plt.show()

# Reason: Pair plots help in visualizing the relationships between different features.

# EDA: Correlation matrix
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Reason: The correlation matrix helps in identifying highly correlated features which might be redundant.

# Create new features
def create_additional_features(df):
    df['price_per_night'] = df['cost'] / df['minimum_nights']
    df['review_frequency'] = df['reviews_per_month'] / df['number_of_reviews']
    df['is_entire_home'] = df['accommodation_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
    
    # New features
    df['log_cost'] = np.log1p(df['cost'])
    df['review_to_cost_ratio'] = df['number_of_reviews'] / df['cost']
    df['high_review_count'] = df['number_of_reviews'].apply(lambda x: 1 if x > 50 else 0)
    city_center = (40.7128, -74.0060)  # Latitude and Longitude of New York City
    df['distance_from_center'] = df.apply(lambda row: geodesic((row['latitude'], row['longitude']), city_center).miles, axis=1)
    
    # Owner related features
    owner_group = df.groupby('owner_id')
    df['total_reviews_per_owner'] = df['owner_id'].map(owner_group['number_of_reviews'].sum())
    df['average_cost_per_owner'] = df['owner_id'].map(owner_group['cost'].mean())
    
    return df

# Apply to training data
train_data = create_additional_features(train_data)

# Separate features and target variable from training data
X = train_data.drop(columns=['yearly_availability'])
y = train_data['yearly_availability']

# Identify numerical and categorical features
numeric_features = ['latitude', 'longitude', 'cost', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'owned_hotels', 'price_per_night', 'review_frequency', 'log_cost', 'review_to_cost_ratio', 'distance_from_center', 'total_reviews_per_owner', 'average_cost_per_owner']
categorical_features = ['region', 'accommodation_type']

# Create transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Hyperparameters for tuning
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 1, 10, 100]
    },
    'SVC': {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    'XGBoost': {
        'classifier__n_estimators': range(10, 60, 100),
        'classifier__max_depth': [3, 6, 9],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.5, 0.7, 0.9],
        'classifier__reg_alpha': [0.01, 0.1, 0.5, 1]
    }
}

best_models = {}
best_scores = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    grid_search = GridSearchCV(pipe, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")

# Save the best models and scores
import pickle
with open('/content/drive/MyDrive/MS/Assessments/Expedia/best_models.pkl', 'wb') as f:
    pickle.dump(best_models, f)
with open('/content/drive/MyDrive/MS/Assessments/Expedia/best_scores.pkl', 'wb') as f:
    pickle.dump(best_scores, f)

# Print classification reports for each model
for name, model in best_models.items():
    y_pred = model.predict(X_val)
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_val, y_pred))
    print("\n")

# Visualize feature importances for the best model (assuming RandomForest is the best model)
best_model = best_models['RandomForest']
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = preprocessor.transformers_[0][2] + list(best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))

# Select top features
num_top_features = 20
top_features = [feature_names[i] for i in indices[:num_top_features]]

print("Top features:")
print(top_features)

# Visualize feature importances
plt.figure(figsize=(10, 8))
plt.title("Feature Importances")
plt.barh(range(num_top_features), importances[indices[:num_top_features]], align='center')
plt.yticks(range(num_top_features), top_features)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.show()

# Load test data
test_data = pd.read_csv('test.csv')

# Create new features in the test data
test_data = create_additional_features(test_data)

# Apply the best model to make predictions on the test set
X_test = test_data.drop(columns=['id'])
predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_data['id'], 'yearly_availability': predictions})
submission.to_csv('submissions.csv', index=False)

print("Submission file created successfully.")
