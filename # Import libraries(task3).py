# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Simulate sample Bank Marketing-like dataset
data = {
    'age': [29, 35, 40, 50, 48, 33, 60, 38, 55, 42],
    'job': ['admin.', 'technician', 'blue-collar', 'management', 'admin.', 'services', 'retired', 'technician', 'management', 'self-employed'],
    'marital': ['single', 'married', 'married', 'married', 'single', 'single', 'married', 'single', 'divorced', 'married'],
    'education': ['secondary', 'tertiary', 'secondary', 'tertiary', 'secondary', 'secondary', 'primary', 'tertiary', 'secondary', 'tertiary'],
    'balance': [1000, 200, 150, 500, 600, 50, 300, 400, 800, 900],
    'housing': ['yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes'],
    'loan': ['no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no'],
    'y': ['no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data)

# Display data
print(df)

# Encode categorical variables
le = LabelEncoder()
for column in ['job', 'marital', 'education', 'housing', 'loan', 'y']:
    df[column] = le.fit_transform(df[column])

# Define features and target
X = df.drop('y', axis=1)
y = df['y']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['No','Yes'], filled=True)
plt.show()
