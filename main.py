import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the training data
train_data = pd.read_csv('train.csv')

# Display first few rows of the dataset to understand its structure
print(train_data.head())

# Check for null values
print(train_data.isnull().sum())

# Handle missing values (e.g., fill with median for numerical, mode for categorical)
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Create new features
def create_features(df):
    df['price_per_night'] = df['cost'] / df['minimum_nights']
    df['review_frequency'] = df['reviews_per_month'] / df['number_of_reviews']
    df['review_frequency'].replace([np.inf, -np.inf], 0, inplace=True)
    df['is_entire_home'] = df['accommodation_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
    return df

# Apply the function to create new features
train_data = create_features(train_data)

# Check for null values again after creating new features
print(train_data.isnull().sum())

# Separate features and target variable from training data
X = train_data.drop(columns=['yearly_availability'])
y = train_data['yearly_availability']

# Identify numerical and categorical features
numeric_features = ['latitude', 'longitude', 'cost', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'owned_hotels', 'price_per_night', 'review_frequency']
categorical_features = ['region']

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
    'SVC': SVC(random_state=42)
}

# Hyperparameters for tuning
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1.0, 10]
    },
    'SVC': {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__kernel': ['linear', 'rbf']
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

# Evaluate the best model on the validation set
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

y_pred = best_model.predict(X_val)
print(f"Validation Accuracy of {best_model_name}: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))

# Extract feature importances (only for models that support it)
if best_model_name == 'RandomForest':
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = numeric_features + list(best_model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
    indices = np.argsort(feature_importances)[::-1]
    
    # Visualize the top 20 features and their importance
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances")
    plt.barh(range(20), feature_importances[indices[:20]], align='center')
    plt.yticks(range(20), [feature_names[i] for i in indices[:20]])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.show()

# Load the test data
test_data = pd.read_csv('test.csv')

# Apply the function to create new features for test data
test_data = create_features(test_data)

# Apply imputers to test data
test_data[numeric_features] = numeric_imputer.transform(test_data[numeric_features])
test_data[categorical_features] = categorical_imputer.transform(test_data[categorical_features])

# Prepare features for the test dataset
X_test = test_data

# Make predictions on the test set using the best model
test_predictions = best_model.predict(X_test)

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'yearly_availability': test_predictions
})

# Save the predictions to a CSV file
submission.to_csv('submissions.csv', index=False)
