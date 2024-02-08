import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diabetes dataset from CSV
diabetes_df = pd.read_csv("diabetes.csv")

# Separate features (X) and target variable (y)
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform grid search with stratified k-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_clf, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the Random Forest classifier with the best hyperparameters
best_rf_clf = RandomForestClassifier(random_state=42, **best_params)
best_rf_clf.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = best_rf_clf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluate using cross-validation
cv_accuracy = cross_val_score(best_rf_clf, X_train_scaled, y_train, cv=stratified_kfold, scoring='accuracy')
print("Cross-Validation Accuracy:", np.mean(cv_accuracy))

# Additional Evaluation Metrics
# Print classification report and confusion matrix for detailed performance analysis
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting accuracy and cross-validation accuracy
plt.figure(figsize=(10, 5))
plt.bar(['Accuracy', 'Cross-Validation Accuracy'], [accuracy, np.mean(cv_accuracy)])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

# Plotting classification report
report = classification_report(y_test, y_pred, output_dict=True)
precisions = [report[str(label)]['precision'] for label in range(2)]
recalls = [report[str(label)]['recall'] for label in range(2)]
f1_scores = [report[str(label)]['f1-score'] for label in range(2)]

labels = ['Non-diabetes', 'Diabetes']
x = range(len(labels))

plt.figure(figsize=(10, 5))
plt.bar(x, precisions, width=0.2, label='Precision')
plt.bar([i + 0.2 for i in x], recalls, width=0.2, label='Recall')
plt.bar([i + 0.4 for i in x], f1_scores, width=0.2, label='F1-Score')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Classification Report Metrics')
plt.xticks([i + 0.1 for i in x], labels)
plt.legend()
plt.show()

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()