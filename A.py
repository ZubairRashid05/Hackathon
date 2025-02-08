import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pandas as pd

dataset_path = pd.read_json("hf://datasets/garyzsu/custom_gym_dataset/train.jsonl", lines=True)

# Load dataset

data = []
with open(dataset_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Assume dataset has 'exercise', 'duration', 'intensity', 'goal' columns
# Encode categorical features
label_encoders = {}
for col in ['exercise', 'goal']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df[['exercise', 'duration', 'intensity']]
y = df['goal']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Function to predict workout routine
def suggest_workout(exercise, duration, intensity):
    if exercise in label_encoders['exercise'].classes_:
        exercise_encoded = label_encoders['exercise'].transform([exercise])[0]
    else:
        raise ValueError("Unknown exercise")

    prediction = model.predict([[exercise_encoded, duration, intensity]])
    goal_decoded = label_encoders['goal'].inverse_transform(prediction)[0]
    return goal_decoded

# Example usage
print(suggest_workout('squat', 30, 7))
