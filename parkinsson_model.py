
# Let's import libraries for data preprocessing and model evaluation
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import f1_score, accuracy_score ,classification_report # Model evaluation metrics
import xgboost as xgb  # XGBoost model
from sklearn.tree import DecisionTreeClassifier
import pickle


# Let's import cardio dataset
df = pd.read_csv('Parkinsson disease.csv')

# Let's remove unnecessary column
del df['name']

#Let's assign target and explanatory values
y = df['status']
X = df[['PPE','MDVP:Fo(Hz)','MDVP:Jitter(%)','MDVP:RAP','D2']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with parameters
clf = DecisionTreeClassifier(max_depth=3 , min_samples_split=2)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = clf.predict(X_train)

# Calculate the accuracy of the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Make predictions on the testing data
y_test_pred = clf.predict(X_test)

# Calculate the accuracy of the model on the testing data
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# new_value = [0.284654,119.992,0.00784,0.00370,2.301442]
# prediction = clf.predict([new_value])
# print(prediction)

# # Save the trained model as a pickle file
# with open('decision_tree_model.pkl', 'wb') as file:
#     pickle.dump(clf, file)
