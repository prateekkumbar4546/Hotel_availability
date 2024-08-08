import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import shap
from scipy import stats

# Load the data
train_data = pd.read_csv('/content/drive/MyDrive/MS/Assessments/Expedia/train.csv')

# Separate features and target variable from training data
X = train_data.drop(columns=['yearly_availability'])
y = train_data['yearly_availability']

# Distribution of the target variable
sns.countplot(x=y)
plt.title('Distribution of Target Variable')
plt.show()

# Check for missing values
missing_values = X.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

# Create new features
X['price_per_night'] = X['cost'] / X['minimum_nights']
X['review_frequency'] = X['reviews_per_month'] / (X['number_of_reviews'] + 1)
X['is_entire_home'] = X['accommodation_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)

# Interaction feature
X['cost_per_review'] = X['cost'] / (X['number_of_reviews'] + 1)

# Identify numerical and categorical features
numeric_features = ['latitude', 'longitude', 'cost', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                    'owned_hotels', 'price_per_night', 'review_frequency', 'cost_per_review']
categorical_features = ['region', 'accommodation_type']

# Check for highly skewed features and apply log transformation
# skewed_features = X[numeric_features].apply(lambda x: stats.skew(x.dropna()))
# skewed_features = skewed_features[skewed_features > 0.85].index
# X[skewed_features] = np.log1p(X[skewed_features])

# Outlier detection and removal using Z-score method
# z_scores = np.abs(stats.zscore(X[numeric_features]))
# X = X[(z_scores <= 3).all(axis=1)]
# y = y.loc[X.index]  # Ensure target variable matches filtered features

print("Data shape after preprocessing:", X.shape)

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

# Apply SMOTE to handle class imbalance
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Hyperparameters for tuning
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10]
    },
    'SVC': {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    'XGBoost': {
        'classifier__n_estimators': range(50, 150, 50),
        'classifier__max_depth': [3, 6, 9],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
best_scores = {}

for name, clf in classifiers.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    grid_search = GridSearchCV(pipe, param_grid[name], cv=skf, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC for {name}: {grid_search.best_score_:.4f}")

# Select the best model based on ROC-AUC score
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

# Evaluate the best model on the validation set
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

print(f"Classification Report for {best_model_name}:\n", classification_report(y_val, y_pred))
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC Score for {best_model_name}: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Feature Importance using SHAP values
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
shap_values = explainer.shap_values(best_model.named_steps['preprocessor'].transform(X_val))

if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # For models that have feature_importances_ attribute
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = best_model.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].get_feature_names_out()
    cat_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    # Print feature importances
    feature_importances = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("Feature Importances:\n", feature_importances)
else:
    print("Feature importance cannot be computed for this model.")
# shap.summary_plot(shap_values, best_model.named_steps['preprocessor'].transform(X_val), feature_names=numeric_features + list(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features)))

# Save predictions on the test dataset
test_data = pd.read_csv('/content/drive/MyDrive/MS/Assessments/Expedia/test (1).csv')
test_data['price_per_night'] = test_data['cost'] / test_data['minimum_nights']
test_data['review_frequency'] = test_data['reviews_per_month'] / (test_data['number_of_reviews'] + 1)
test_data['is_entire_home'] = test_data['accommodation_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
test_data['cost_per_review'] = test_data['cost'] / (test_data['number_of_reviews'] + 1)

X_test = test_data.drop(columns=['id'])
# X_test[skewed_features] = np.log1p(X_test[skewed_features])

# Apply the preprocessing and prediction pipeline
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
test_predictions = best_model.named_steps['classifier'].predict(X_test_transformed)

submission = pd.DataFrame({'id': test_data['id'], 'yearly_availability': test_predictions})
submission.to_csv('submission.csv', index=False)
