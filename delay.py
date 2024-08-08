import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the data with features
train_data = pd.read_csv('train.csv')

# EDA: Checking null values
print(train_data.isnull().sum())

# EDA: Distribution of columns
print(train_data.describe())

# EDA: Visualize pair plots
sns.pairplot(train_data)
plt.show()

# EDA: Correlation matrix
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Create new features
def create_additional_features(df):
    df['total_flight_time'] = df['SCHEDULED_ARRIVAL'] - df['SCHEDULED_DEPARTURE']
    df['departure_hour'] = df['SCHEDULED_DEPARTURE'] // 100
    df['arrival_hour'] = df['SCHEDULED_ARRIVAL'] // 100
    return df

# Apply to training data
train_data = create_additional_features(train_data)

# Separate features and target variable from training data
X = train_data.drop(columns=['ARRIVAL_DELAY'])
y = train_data['ARRIVAL_DELAY']

# Identify numerical and categorical features
numeric_features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 
                    'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'DISTANCE', 'total_flight_time', 
                    'departure_hour', 'arrival_hour']
categorical_features = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']

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

# Define regressors
regressors = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Hyperparameters for tuning
param_grid = {
    'RandomForest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20]
    },
    'SVR': {
        'regressor__C': [0.1, 1.0, 10],
        'regressor__kernel': ['linear', 'rbf']
    },
    'XGBoost': {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [3, 6, 9],
        'regressor__learning_rate': [0.01, 0.1, 0.2]
    }
}

best_models = {}
best_scores = {}

# Train and evaluate each regressor
for name, reg in regressors.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    
    grid_search = GridSearchCV(pipe, param_grid[name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = np.sqrt(-grid_search.best_score_)  # RMSE
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation RMSE for {name}: {np.sqrt(-grid_search.best_score_):.4f}")

# Save the best models and scores
import pickle
with open('best_models.pkl', 'wb') as f:
    pickle.dump(best_models, f)
with open('best_scores.pkl', 'wb') as f:
    pickle.dump(best_scores, f)

# Predict and evaluate on validation set
for name, model in best_models.items():
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE for {name} on validation set: {rmse:.4f}")

    # Feature importance visualization for RandomForest and XGBoost
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        feature_names = (model.named_steps['preprocessor']
                         .transformers_[0][1]
                         .named_steps['scaler']
                         .transform(X_train.select_dtypes(include=np.number).columns)
                         .tolist()
                         + model.named_steps['preprocessor']
                         .transformers_[1][1]
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features))
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importance for {name}')
        plt.show()

# Load test data and apply best model for predictions
test_data = pd.read_csv('test.csv')
test_data = create_additional_features(test_data)
X_test = test_data.drop(columns=['id'])
predictions = best_models['RandomForest'].predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_data['id'], 'arrival_delay': predictions})
submission.to_csv('submissions.csv', index=False)

print("Submission file created successfully.")
