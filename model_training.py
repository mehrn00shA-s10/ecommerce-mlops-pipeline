import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from feature_engineering import file_path


file_path = os.path.join('data', 'session_features.csv')
df = pd.read_csv(file_path)

x = df.drop(columns=['user_session', 'is_purchased'])
y = df['is_purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

print(f"training set size: {x_train.shape[0]}")
print(f"testing set size: {x_test.shape[0]}")

# model A: logistic regression
print('A: logistic regression')
model_A = LogisticRegression(max_iter=1000)
model_A.fit(x_train, y_train)

# evaluate A:
yhat_A = model_A.predict(x_test)
acc_a = accuracy_score(y_test, yhat_A)
print(f"Model A Accuracy: {acc_a:.4f}")
print("\n--- Model A Classification Report ---")
print(classification_report(y_test, yhat_A))

# model B:
print('B: random forest')
model_B  = RandomForestClassifier(n_estimators=100, random_state=42)
model_B.fit(x_train, y_train)

# evaluate Model B
yhat_B = model_B.predict(x_test)
acc_b = accuracy_score(y_test, yhat_B)
print(f"Model B Accuracy: {acc_b:.4f}")

print("\n--- Model B Classification Report ---")
print(classification_report(y_test, yhat_B))


models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, 'model_a.pkl'), 'wb') as f:
    pickle.dump(model_A, f)

with open(os.path.join(models_dir, 'model_b.pkl'), 'wb') as f:
    pickle.dump(model_B, f)

print("\n✓ Both models successfully trained and saved in the 'models/' directory!")