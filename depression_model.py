import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib
import warnings
import xgboost as xgb
import lightgbm as lgb
import random
import shap
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
from sklearn.metrics import precision_score
import fairlearn.metrics
random.seed(42)
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set the number of cores to use for joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust this number based on your system

# Create graphics directory if it doesn't exist
os.makedirs('graphics', exist_ok=True)
# Create pre_process directory if it doesn't exist
os.makedirs('pre_process', exist_ok=True)

# Load the raw data
raw_data = pd.read_csv('student_depression_dataset.csv')

# Function to sanitize column names for file names
def sanitize_filename(name):
    return "".join([c if c.isalnum() else "_" for c in name])

# Plot all columns in the CSV file
# for column in raw_data.columns:
#     plt.figure(figsize=(8, 6))
#     if raw_data[column].dtype == 'object' or raw_data[column].nunique() < 20:
#         # Categorical data
#         sns.countplot(x=column, data=raw_data)
#         plt.title(f'Distribution of {column} (Raw Data)')
#         plt.xlabel(column)
#         plt.ylabel('Count')
#     else:
#         # Numerical data
#         sns.histplot(raw_data[column], bins=20, kde=True)
#         plt.title(f'Distribution of {column} (Raw Data)')
#         plt.xlabel(column)
#         plt.ylabel('Frequency')
    
#     # Save the plot with sanitized column name
#     sanitized_column = sanitize_filename(column)
#     plt.savefig(f'pre_process/{sanitized_column}_distribution_raw.png')
#     plt.show()
#     plt.close()

# Load the raw data
raw_data = pd.read_csv('student_depression_dataset.csv')

# # Distribution of Depression
# plt.figure(figsize=(8, 6))
# sns.countplot(x='Depression', data=raw_data)
# plt.title('Distribution of Depression (Raw Data)')
# plt.xlabel('Depression')
# plt.ylabel('Count')
# plt.savefig('pre_process/distribution_of_depression_raw.png')
# plt.show()
# plt.close()

# Gender vs Depression
# plt.figure(figsize=(8, 6))
# sns.countplot(x='Gender', hue='Depression', data=raw_data)
# plt.title('Gender vs Depression (Raw Data)')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.savefig('pre_process/gender_vs_depression_raw.png')
# plt.show()
# plt.close()

# # Age Distribution
# plt.figure(figsize=(8, 6))
# sns.histplot(raw_data['Age'], bins=20, kde=True)
# plt.title('Age Distribution (Raw Data)')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig('pre_process/age_distribution_raw.png')
# plt.show()
# plt.close()

# # CGPA vs Depression
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Depression', y='CGPA', data=raw_data)
# plt.title('CGPA vs Depression (Raw Data)')
# plt.xlabel('Depression')
# plt.ylabel('CGPA')
# plt.savefig('pre_process/cgpa_vs_depression_raw.png')
# plt.show()
# plt.close()

# # Sleep Duration vs Depression
# plt.figure(figsize=(12, 6))
# sns.countplot(x='Sleep Duration', hue='Depression', data=raw_data)
# plt.title('Sleep Duration vs Depression (Raw Data)')
# plt.xlabel('Sleep Duration')
# plt.ylabel('Count')
# plt.savefig('pre_process/sleep_duration_vs_depression_raw.png')
# plt.show()
# plt.close()

# # Correlation Heatmap
# plt.figure(figsize=(12, 10))
# numeric_df = raw_data.select_dtypes(include=[np.number])
# correlation_matrix = numeric_df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap (Raw Data)')
# plt.savefig('pre_process/correlation_heatmap_raw.png')
# plt.show()
# plt.close()

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the DataPreprocessor with the dataset path.
        
        Parameters:
        file_path (str): Path to the CSV dataset file.
        """
        self.file_path = file_path
        self.data = None
        self.encoder = None
        self.label_encoder_new_degree = None
        self.label_encoder_depression = None

    def load_data(self):
        """Load the dataset from the file path."""
        self.data = pd.read_csv(self.file_path)
        print("Data loaded successfully.")

    def preprocess_data(self):
        """
        Perform preprocessing on the loaded dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call `load_data()` first.")
    
        # Drop unnecessary columns
        self.data = self.data.drop(['id'], axis=1)
    
        # Convert Gender to numeric
        self.data['Gender'] = self.data['Gender'].map({'Male': 0, 'Female': 1})
    
        # Filter Cities with fewer than 450 students
        city_counts = self.data['City'].value_counts()
        self.data = self.data[self.data['City'].isin(city_counts[city_counts >= 450].index)]
    
        # Keep only 'Student' in Profession column
        self.data = self.data[self.data['Profession'] == 'Student'].drop(['Profession'], axis=1)
    
        # Drop irrelevant columns
        self.data = self.data.drop(['Work Pressure'], axis=1)
        self.data = self.data[self.data['Age'] <= 34]
        self.data = self.data[self.data['Academic Pressure'] > 0]
        self.data = self.data[self.data['Study Satisfaction'] > 0]
        self.data = self.data.drop(['Job Satisfaction'], axis=1)
    
        # Process Sleep Duration
        sleep_map = {
            'Less than 5 hours': 0,
            '5-6 hours': 1,
            '7-8 hours': 2,
            'More than 8 hours': 3
        }
        self.data = self.data[self.data['Sleep Duration'] != 'Others']
        self.data['Sleep Duration'] = self.data['Sleep Duration'].map(sleep_map)
    
        # Process Dietary Habits
        diet_map = {'Healthy': 0, 'Unhealthy': 1, 'Moderate': 2}
        self.data = self.data[self.data['Dietary Habits'] != 'Others']
        self.data['Dietary Habits'] = self.data['Dietary Habits'].map(diet_map)
    
        # Map Degree to New Degree Categories
        degree_mapping = {
            r'BSc|BCA|B.Ed|BHM|B.Pharm|B.Com|BE|BA|B.Arch|B.Tech|BBA|LLB': 'Đã tốt nghiệp',
            r'MSc|MCA|M.Ed|M.Pharm|M.Com|ME|MA|M.Arch|M.Tech|MBA|LLM': 'Sau khi tốt nghiệp',
            'Class 12': 'Trung học phổ thông'
        }
        
        for pattern, category in degree_mapping.items():
            self.data.loc[self.data['Degree'].str.contains(pattern, regex=True, na=False), 'New_Degree'] = category
        
        self.data = self.data[self.data['Degree'] != 'Others']
        new_degree_map = {'Đã tốt nghiệp': 0, 'Sau khi tốt nghiệp': 1, 'Trung học phổ thông': 2}
        self.data['New_Degree'] = self.data['New_Degree'].map(new_degree_map)
    
        # Drop the original Degree column
        self.data = self.data.drop(['Degree'], axis=1)
    
        # Map Suicidal Thoughts and Family History
        self.data['Have you ever had suicidal thoughts ?'] = self.data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
        self.data['Family History of Mental Illness'] = self.data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
    
        # Encode Depression
        self.label_encoder_depression = LabelEncoder()
        self.data['Depression'] = self.label_encoder_depression.fit_transform(self.data['Depression'])
    
        # Encode City
        self.data['City_encoded'] = self.data['City'].astype('category').cat.codes
    
        # Drop the original City column
        self.data = self.data.drop(['City'], axis=1)
    
        # Cluster CGPA into specified ranges and overwrite the original CGPA column
        bins = [0, 5, 6, 7, 8, 9, 10]
        labels = ['<5', '5 - <6', '6 - <7', '7 - <8', '8 - <9', '9 - 10']
        self.data['CGPA'] = pd.cut(self.data['CGPA'], bins=bins, labels=labels, right=False)
    
        # Convert CGPA labels to numerical values
        cgpa_map = {'<5': 0, '5 - <6': 1, '6 - <7': 2, '7 - <8': 3, '8 - <9': 4, '9 - 10': 5}
        self.data['CGPA'] = self.data['CGPA'].map(cgpa_map)
    
        # Drop rows with missing values
        self.data = self.data.dropna()
    
        print("Preprocessing complete.")
    
    def save_data(self, output_path):
        """
        Save the preprocessed dataset to a CSV file.
        
        Parameters:
        output_path (str): Path to save the processed dataset.
        """
        if self.data is None:
            raise ValueError("No data to save. Please preprocess the data first.")
    
        self.data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}.")

# Example usage
preprocessor = DataPreprocessor(file_path='student_depression_dataset.csv')
preprocessor.load_data()
preprocessor.preprocess_data()
preprocessor.save_data(output_path='Processed_StudentDepression.csv')

# Load the cleaned data
df = pd.read_csv('Processed_StudentDepression.csv')

# Define numerical features
numerical_features = [
    'Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
    'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
    'Financial Stress', 'Family History of Mental Illness', 'New_Degree', 'City_encoded'
]

# Normalize/scale numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the data into training and testing sets
X = df.drop(['Depression'], axis=1)
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the split data
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)

# Save the split data to CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Model training and evaluation with GridSearchCV
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1)
}

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "saga"],
        "max_iter": [100, 200, 300]
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [3, 5, 10]
    },
    "SVM": {
        "C": [0.01, 0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"]
    },
    "Naive Bayes": {},
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [3, 5, 10]
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [3, 5, 10]
    }
}


best_model_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    best_model_results[name] = {
        "best_model": best_model,
        "accuracy": accuracy,
        "best_params": grid_search.best_params_
    }
    print(f"{name} - Best Accuracy: {accuracy:.4f}")

# Find the best model
best_model_name = max(best_model_results, key=lambda k: best_model_results[k]['accuracy'])
best_model = best_model_results[best_model_name]['best_model']
print(f"Best Model: {best_model_name} with accuracy {best_model_results[best_model_name]['accuracy']:.4f}")

# Save the best model and the encoders
joblib.dump(best_model, 'depression_model_best.pkl')
joblib.dump(scaler, 'scaler_best.pkl')
joblib.dump(preprocessor.label_encoder_new_degree, 'label_encoder_new_degree_best.pkl')
joblib.dump(preprocessor.label_encoder_depression, 'label_encoder_depression_best.pkl')
print("Best model and encoders saved.")

# Data Visualization

# Distribution of Depression
plt.figure(figsize=(8, 6))
sns.countplot(x='Depression', data=df)
plt.title('Distribution of Depression')
plt.xlabel('Depression')
plt.ylabel('Count')
plt.savefig('graphics/distribution_of_depression.png')
plt.show()
plt.close()

# Gender vs Depression
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Depression', data=df)
plt.title('Gender vs Depression')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('graphics/gender_vs_depression.png')
plt.show()
plt.close()

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('graphics/age_distribution.png')
plt.show()
plt.close()

# CGPA vs Depression
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='CGPA', data=df)
plt.title('CGPA vs Depression')
plt.xlabel('Depression')
plt.ylabel('CGPA')
plt.savefig('graphics/cgpa_vs_depression.png')
plt.show()
plt.close()

# Sleep Duration vs Depression
plt.figure(figsize=(12, 6))
sns.countplot(x='Sleep Duration', hue='Depression', data=df)
plt.title('Sleep Duration vs Depression')
plt.xlabel('Sleep Duration')
plt.ylabel('Count')
plt.savefig('graphics/sleep_duration_vs_depression.png')
plt.show()
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('graphics/correlation_heatmap.png')
plt.show()
plt.close()

# Plot confusion matrix for all models
for name, result in best_model_results.items():
    model = result['best_model']
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: 0', 'Predicted: 1'], yticklabels=['Real: 0', 'Real: 1'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Real Values')
    if name == best_model_name:
        plt.title(f'Confusion Matrix - {name} (Best Model)')
        plt.savefig(f'graphics/confusion_matrix_{name}_best.png')
    else:
        plt.savefig(f'graphics/confusion_matrix_{name}.png')
    plt.show()
    plt.close()

# Plot ROC Curves for top-3 models
from sklearn.metrics import roc_curve, auc

# Sort models by accuracy
best_model_results_sorted = dict(sorted(best_model_results.items(), key=lambda item: item[1]['accuracy'], reverse=True))

# Get the top 3 models from GridSearchCV
top_3_models = list(best_model_results_sorted.keys())[:3]

# Initialize a figure for plotting ROC curves
plt.figure(figsize=(16, 8))

# Iterate over the top 3 models
for model_name in top_3_models:
    model = best_model_results[model_name]['best_model']
    
    # Get predictions probabilities (for ROC curve, need probability scores)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class
    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Compute the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Labels and title
plt.title('ROC Curves for Top 3 Models', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)

# Save the plot
plt.savefig('graphics/top_3_models_roc_curves.png', bbox_inches='tight', dpi=300)

# Display the plot
plt.show()

# Plotting the results
plt.figure(figsize=(15, 6))
sns.barplot(
    x=[result["accuracy"] for result in best_model_results_sorted.values()],
    y=list(best_model_results_sorted.keys()),
    palette='Blues'
)

plt.xlabel('Accuracy', fontsize=14)
plt.title('Grid Search Model Comparison', fontsize=18)

# Increase the tick label font sizes, including model names
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)  # Adjust font size of the model names

# Add accuracy labels
for i, (name, result) in enumerate(best_model_results_sorted.items()):
    plt.text(result["accuracy"], i, f'{result["accuracy"]*100:.2f}%', color='black', va='center', fontsize=14)

# Save the plot to a file
plt.savefig('graphics/mental_health_model_comparison_grid_search.png', bbox_inches='tight', dpi=300)
plt.show()

# Perform XAI on the top-performing model(s)
# Feature Importance Analysis
model_names_str = list(models.keys())
print(model_names_str)
model_name = "XGBoost"
best_params = best_model_results[model_name]['best_params']

# Get feature names
feature_names = df.drop('Depression', axis=1).columns

# Initialize the XGBoost model with best parameters
xgb_model = xgb.XGBClassifier(random_state=42, **best_params)

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy*100:.2f}%")

# Visualize the feature importance
plt.figure(figsize=(12, 8))

# Get feature importances
importance = xgb_model.feature_importances_

# Sort features by importance
indices = np.argsort(importance)[::-1]

# Select top 10 features
top_features = indices[:10]

# Create a horizontal bar plot
plt.barh(range(len(top_features)), importance[top_features], color='skyblue')

# Set y-axis labels to actual feature names
plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])

plt.title("XGBoost Feature Importance", fontsize=18)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)

plt.tight_layout()
plt.grid()
# Save the plot as a PNG image
plt.savefig('graphics/xgboost_feature_importance_horizontal.png', bbox_inches='tight', dpi=300)
plt.show()

# Optional: Print out the top 10 features and their importance scores
for f in range(len(top_features)):
    print("%d. %s: %f" % (f + 1, feature_names[top_features[f]], importance[top_features[f]]))
# Define model names and best parameters
model_names_str = list(models.keys())
print(model_names_str)
# Gradient Boosting Feature Importance
model_name = "Gradient Boosting"
best_params = best_model_results[model_name]['best_params']
# Initialize the Gradient Boosting model with best parameters
gb_model = GradientBoostingClassifier(random_state=42, **best_params)
# Train the model
gb_model.fit(X_train, y_train)
# Evaluate the model
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Accuracy: {accuracy*100:.2f}%")
# Visualize the feature importance
plt.figure(figsize=(12, 8))
# Get feature importances
importance = gb_model.feature_importances_
# Sort features by importance
indices = np.argsort(importance)[::-1]
# Select top 10 features
top_features = indices[:10]
# Create a horizontal bar plot
plt.barh(range(len(top_features)), importance[top_features], color='lightgreen')
# Set y-axis labels to actual feature names
plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
plt.title("Gradient Boosting Feature Importance", fontsize=18)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.grid()
# Save the plot as a PNG image
plt.savefig('graphics/gradient_boosting_feature_importance_horizontal.png', bbox_inches='tight', dpi=300)
plt.show()
# Optional: Print out the top 10 features and their importance scores
for f in range(len(top_features)):
    print("%d. %s: %f" % (f + 1, feature_names[top_features[f]], importance[top_features[f]]))
# LightGBM Feature Importance
model_name = "LightGBM"
best_params = best_model_results[model_name]['best_params']
# Initialize the LightGBM model with best parameters
lgb_model = lgb.LGBMClassifier(random_state=42, **best_params)
# Train the model
lgb_model.fit(X_train, y_train)
# Evaluate the model
y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {accuracy*100:.2f}%")
# Visualize the feature importance
plt.figure(figsize=(12, 8))
# Get feature importances
importance = lgb_model.feature_importances_
# Sort features by importance
indices = np.argsort(importance)[::-1]
# Select top 10 features
top_features = indices[:10]
# Create a horizontal bar plot
plt.barh(range(len(top_features)), importance[top_features], color='lightcoral')
# Set y-axis labels to actual feature names
plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
plt.title("LightGBM Feature Importance", fontsize=18)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.grid()
# Save the plot as a PNG image
plt.savefig('graphics/lightgbm_feature_importance_horizontal.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()
# # Optional: Print out the top 10 features and their importance scores
# for f in range(len(top_features)):
#     print("%d. %s: %f" % (f + 1, feature_names[top_features[f]], importance[top_features[f]]))

def get_best_params(model_name, best_model_results):
    return best_model_results[model_name]['best_params']

# Initialize the models with the best parameters from GridSearchCV
xgb_best_params = get_best_params("XGBoost", best_model_results)
gb_best_params = get_best_params("Gradient Boosting", best_model_results)
lgb_best_params = get_best_params("LightGBM", best_model_results)

# Initialize the models with best hyperparameters
xgb_model = xgb.XGBClassifier(random_state=42, **xgb_best_params)
gb_model = GradientBoostingClassifier(random_state=42, **gb_best_params)
lgb_model = lgb.LGBMClassifier(random_state=42, **lgb_best_params)

# Train the models
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Evaluate the models (for reference)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)

print(f"XGBoost Accuracy: {accuracy_xgb*100:.2f}%")
print(f"Gradient Boosting Accuracy: {accuracy_gb*100:.2f}%")
print(f"LightGBM Accuracy: {accuracy_lgb*100:.2f}%")

# Get feature names (excluding the target column)
feature_names = df.drop('Depression', axis=1).columns

# Get the feature importances for each model
importance_xgb = xgb_model.feature_importances_
importance_gb = gb_model.feature_importances_
importance_lgb = lgb_model.feature_importances_

# Sort features by importance for each model
indices_xgb = np.argsort(importance_xgb)[::-1]
indices_gb = np.argsort(importance_gb)[::-1]
indices_lgb = np.argsort(importance_lgb)[::-1]

# Select top 10 features (or as many as needed)
top_n = 10
top_features_xgb = indices_xgb[:top_n]
top_features_gb = indices_gb[:top_n]
top_features_lgb = indices_lgb[:top_n]

# Collect the importance values for each model (top features)
importance_data = {
    "Feature": [feature_names[i] for i in top_features_xgb],
    "XGBoost": importance_xgb[top_features_xgb],
    "Gradient Boosting": importance_gb[top_features_gb],
    "LightGBM": importance_lgb[top_features_lgb] / 100
}

# Convert the data to a DataFrame for easier plotting
importance_df = pd.DataFrame(importance_data)

# Plot the grouped bar plot
plt.figure(figsize=(14, 8))
importance_df.set_index("Feature").plot(kind='bar', width=0.8, color=['skyblue', 'lightgreen', 'salmon'], figsize=(14, 8))

plt.title("Top 10 Feature Importances Across Models", fontsize=18)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Importance", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)

# Save the plot as a PNG image
plt.savefig('graphics/model_comparison_feature_importance.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()  # Close the figure to avoid displaying it multiple times

# Create a SHAP explainer object
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Plot a summary plot for SHAP values
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Optional: Create a force plot for a specific prediction
shap.initjs()  # to display JS visualizations in notebooks

plt.figure(figsize=(10, 8))
fig = shap.force_plot(shap_values[0], show=False)  # Visualize the first test sample
# plt.savefig('graphics/shap_original_summary_plot.png', bbox_inches='tight', dpi=300)
plt.show()

# Save the force plot as an HTML file
shap.save_html('graphics/shap_original_summary_plot.html', fig)

# Create a LIME explainer for the dataset
explainer = LimeTabularExplainer(X_train.values, training_labels=y_train.values, mode='classification', feature_names=feature_names)

# Explain a single prediction
exp = explainer.explain_instance(X_test.values[0], xgb_model.predict_proba, num_features=10)

plt.figure(figsize=(10, 8))
# Plot the explanation
fig = exp.as_pyplot_figure()
plt.savefig('graphics/lime_tabular_summary_plot.png', bbox_inches='tight', dpi=300)
plt.show()

# TreeSHAP explanation for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
fig = shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig('graphics/shap_summary_plot.png', bbox_inches='tight', dpi=300)
plt.show()

global_feature_names = feature_names
# Get feature names (excluding the target column)
feature_names = df.drop('Depression', axis=1).columns
global_feature_names = feature_names

# Define the models with best hyperparameters from GridSearchCV
xgb_best_params = get_best_params("XGBoost", best_model_results)
gb_best_params = get_best_params("Gradient Boosting", best_model_results)
lgb_best_params = get_best_params("LightGBM", best_model_results)

# Initialize the models with best hyperparameters
xgb_model = xgb.XGBClassifier(random_state=42, **xgb_best_params)
gb_model = GradientBoostingClassifier(random_state=42, **gb_best_params)
lgb_model = lgb.LGBMClassifier(random_state=42, **lgb_best_params)

# List of models to evaluate
models = {
    "XGBoost": xgb_model,
    "Gradient Boosting": gb_model,
    "LightGBM": lgb_model
}

# Train the models
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Get predictions for each model
y_pred_xgb = xgb_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)

y_preds = {
    "XGBoost": y_pred_xgb,
    "Gradient Boosting": y_pred_gb,
    "LightGBM": y_pred_lgb
}

# Function to compute fairness metrics for a given model and sensitive feature
def compute_fairness_metrics(y_true, y_pred, sensitive_features, sensitive_column_name):
    # Compute fairness metrics for each sensitive feature
    fairness_metrics = {}
    
    fairness_metrics['EOR'] = fairlearn.metrics.equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    fairness_metrics['DPD'] = fairlearn.metrics.demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    fairness_metrics['DPR'] = fairlearn.metrics.demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    
    return fairness_metrics

# Iterate over models and sensitive columns to compute fairness metrics
sensitive_columns = X.columns.tolist()  # All feature column names
results = []

for model_name, model in models.items():
    # Get predictions for the model
    y_pred = y_preds[model_name]
    
    for column in sensitive_columns:
        # Get sensitive feature (assumed to be in X_test)
        sensitive_column_index = X.columns.tolist().index(column)
        sensitive_features = X_test.iloc[:, sensitive_column_index]
        
        # Compute fairness metrics
        fairness_metrics = compute_fairness_metrics(y_test, y_pred, sensitive_features, column)
        
        # Store results for the current model and sensitive feature
        row = {
            "Model": model_name,
            "Sensitive Feature": column,
            "EOR": fairness_metrics['EOR'],
            "DPD": fairness_metrics['DPD'],
            "DPR": fairness_metrics['DPR']
        }
        results.append(row)

# Create DataFrame to display results in tabular format
df_results = pd.DataFrame(results)

# Display the results as a table
print(df_results)

# Function to get top features
def get_top_features(model, X_train, n_top=10):
    # Check the type of model and use appropriate method
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (XGBoost, Random Forest, etc.)
        feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'{global_feature_names[i]}' for i in range(X_train.shape[1])]
        
        feature_importances = model.feature_importances_
        top_indices = np.argsort(feature_importances)[::-1][:n_top]
        
        top_features = X_train.iloc[:, top_indices] if isinstance(X_train, pd.DataFrame) else X_train[:, top_indices]
        top_feature_names = [global_feature_names[i] for i in top_indices]
        
        return top_features, top_indices, top_feature_names
    
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression, etc.)
        feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'{global_feature_names[i]}' for i in range(X_train.shape[1])]
        
        feature_importances = np.abs(model.coef_[0])
        top_indices = np.argsort(feature_importances)[::-1][:n_top]
        
        top_features = X_train.iloc[:, top_indices] if isinstance(X_train, pd.DataFrame) else X_train[:, top_indices]
        top_feature_names = [global_feature_names[i] for i in top_indices]
        
        return top_features, top_indices, top_feature_names
    
    else:
        raise ValueError("Model type not supported for feature importance")

# Prepare results for visualization
results_list = []
for model_name, model in models.items():
    # Get predictions for the model
    y_pred = y_preds[model_name]
    
    # Get top 10 features
    try:
        _, top_indices, top_feature_names = get_top_features(model, X_train)
    except ValueError:
        print(f"Skipping feature importance for {model_name}")
        continue
    
    for i, feature_name in enumerate(top_feature_names):
        # Get sensitive feature (assumed to be in X_test)
        sensitive_column_index = top_indices[i]
        sensitive_features = X_test.iloc[:, sensitive_column_index]
        
        # Compute fairness metrics
        fairness_metrics = compute_fairness_metrics(y_test, y_pred, sensitive_features, feature_name)
        
        # Add to results list
        for metric_name, metric_value in fairness_metrics.items():
            results_list.append({
                'Model': model_name,
                'Sensitive Feature': feature_name,
                'Metric': metric_name,
                'Value': metric_value
            })

# Convert to DataFrame
df_results = pd.DataFrame(results_list)

# Plotting
# Set up a figure with multiple subplots for each fairness metric
fig, axes = plt.subplots(1, 3, figsize=(18, 8))  # Adjust figure size for more space

# Bar plot for each fairness metric
metrics = ['EOR', 'DPD', 'DPR']
for idx, metric in enumerate(metrics):
    # Filter data for specific metric
    metric_data = df_results[df_results['Metric'] == metric]
    
    ax = axes[idx]
    sns.barplot(x='Sensitive Feature', y='Value', hue='Model', data=metric_data, ax=ax)
    
    ax.set_title(f'{metric} by Model and Sensitive Feature', fontsize=18)
    ax.set_xlabel('Sensitive Feature', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    
    # Rotate the x-axis labels for better readability
    ax.set_xticks(range(len(ax.get_xticklabels())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    # Increase space between labels and plot
    ax.tick_params(axis='x', pad=10)

# Adjust layout to increase space between subplots and prevent overlap
plt.tight_layout(w_pad=3, h_pad=3)  # Increase horizontal and vertical padding
plt.savefig('graphics/fairness_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Set up a figure with multiple subplots for each fairness metric
fig, axes = plt.subplots(1, 3, figsize=(18, 8))  # Adjust figure size for more space

# Bar plot for each fairness metric
for idx, metric in enumerate(['EOR', 'DPD', 'DPR']):
    ax = axes[idx]
    sns.barplot(x='Sensitive Feature', y=metric, hue='Model', data=df_results, ax=ax)
    ax.set_title(f'{metric} by Model and Sensitive Feature', fontsize=14)
    ax.set_xlabel('Sensitive Feature', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    
    # Rotate the x-axis labels for better readability
    ax.set_xticks(range(len(ax.get_xticklabels())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Increase space between labels and plot
    ax.tick_params(axis='x', pad=10)

# Adjust layout to increase space between subplots and prevent overlap
plt.tight_layout(w_pad=3, h_pad=3)  # Increase horizontal and vertical padding
plt.savefig('graphics/fairness_metrics_all_features.png', dpi=300, bbox_inches='tight')
plt.show()