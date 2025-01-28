import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing classes

# Dataset
df = pd.read_csv('Phishing_Dataset.csv')

# Converts categorical columns to numeric (if applicable)
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Features and target variables
X = df.drop('status', axis=1)
y = df['status']

# Standardizing the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 1. Random Forest with SMOTE (Class Imbalance Handling)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Cross-validation score for Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
start_time = time.time()  # Start time measurement for Random Forest training
rf_model.fit(X_train_smote, y_train_smote)
rf_training_time = time.time() - start_time  # End time measurement

rf_cross_val_score = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy').mean()

# 3. Logistic Regression for comparison
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
start_time = time.time()  # Start time measurement for Logistic Regression training
log_reg_model.fit(X_train, y_train)
log_reg_training_time = time.time() - start_time  # End time measurement

# Predictions
rf_predictions = rf_model.predict(X_test)
log_reg_predictions = log_reg_model.predict(X_test)

# Metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions)

# Metrics for Logistic Regression
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
log_reg_precision = precision_score(y_test, log_reg_predictions)
log_reg_recall = recall_score(y_test, log_reg_predictions)
log_reg_f1_score = f1_score(y_test, log_reg_predictions)

# ROC Curve for both models
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_lr, tpr_lr, _ = roc_curve(y_test, log_reg_model.predict_proba(X_test)[:, 1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
log_reg_cm = confusion_matrix(y_test, log_reg_predictions)

# Plotting
plt.figure(figsize=(12, 4))

# Confusion Matrices 
plt.subplot(1, 2, 1) 
plt.title("Random Forest Confusion Matrix")
plt.imshow(rf_cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)  
plt.title("Logistic Regression Confusion Matrix")
plt.imshow(log_reg_cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
plt.title('ROC Curve Comparison')
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, color='red', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Bar Graph of Model Metrics 
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [rf_accuracy, log_reg_accuracy],
    'Precision': [rf_precision, log_reg_precision],
    'Recall': [rf_recall, log_reg_recall],
    'F1-Score': [rf_f1_score, log_reg_f1_score]
})

metrics_df.set_index('Model', inplace=True)

# Plot bar graph of metrics
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Comparison Metrics")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Displaying the results with time taken for training
print("\nModel Comparison Metrics:")
print(metrics_df)

print("\nTraining Times:")
print(f"Random Forest Training Time: {rf_training_time:.4f} seconds")
print(f"Logistic Regression Training Time: {log_reg_training_time:.4f} seconds")