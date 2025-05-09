import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Load Dataset ---
# Make sure this CSV exists or simulate with your own
try:
    df = pd.read_csv('traffic_accidents.csv')
except FileNotFoundError:
    print("Sample data not found. Creating a dummy dataset.")
    df = pd.DataFrame({
        'Time': np.random.randint(0, 24, 100),
        'Location': np.random.choice(['Downtown', 'Suburb', 'Highway'], 100),
        'Weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], 100),
        'Vehicles_Involved': np.random.randint(1, 5, 100),
        'Severity': np.random.choice(['Low', 'Medium', 'High'], 100)
    })

# --- 2. Preprocessing ---
label_encoders = {}
for col in ['Location', 'Weather', 'Severity']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df = df.dropna()

X = df[['Time', 'Location', 'Weather', 'Vehicles_Involved']]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluate Model ---
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoders['Severity'].classes_))

# --- 5. Predict on New Data ---
new_event = pd.DataFrame([[14, label_encoders['Location'].transform(['Downtown'])[0],
                           label_encoders['Weather'].transform(['Rain'])[0], 2]],
                         columns=X.columns)

prediction = model.predict(new_event)
predicted_label = label_encoders['Severity'].inverse_transform(prediction)
print("Predicted Severity for new event:", predicted_label[0])

# --- 6. Visualizations ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x='Severity', data=df)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity (Encoded)")
plt.ylabel("Count")
plt.show()
