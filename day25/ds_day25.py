import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = [
    [53, "Male", "No", "Smartphone", 2, "Credit Card", 5538.33, 791.19, 7, "Standard", "Cancelled"],
    [53, "Male", "No", "Tablet", 3, "Paypal", 741.09, 247.03, 3, "Overnight", "Completed"],
    [41, "Male", "No", "Laptop", 3, "Credit Card", 1855.84, 463.96, 4, "Express", "Completed"],
    [41, "Male", "Yes", "Smartphone", 2, "Cash", 3164.76, 791.19, 4, "Overnight", "Completed"],
    [75, "Male", "Yes", "Smartphone", 5, "Cash", 41.5, 20.75, 2, "Express", "Completed"],
    [30, "Female", "No", "Laptop", 1, "Cash", 5000, 1000, 5, "Standard", "Cancelled"],
    [25, "Male", "Yes", "Tablet", 4, "Credit Card", 1200, 300, 4, "Express", "Completed"],
    [60, "Female", "No", "Laptop", 2, "Cash", 4800, 1200, 4, "Standard", "Cancelled"],
    [35, "Male", "Yes", "Smartphone", 5, "Paypal", 2000, 500, 4, "Express", "Completed"],
    [28, "Female", "No", "Tablet", 3, "Credit Card", 900, 300, 3, "Standard", "Cancelled"],
]

columns = [
    "Age", "Gender", "Loyalty Member", "Product Type",
    "Rating", "Payment Method", "Total Price",
    "Unit Price", "Quantity", "Shipping Type", "Order Status"
]

df = pd.DataFrame(data, columns=columns)

print("Dataset:\n", df)

df['Order Status'] = df['Order Status'].map({'Cancelled': 0, 'Completed': 1})

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Order Status', axis=1)
y = df['Order Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_customer(input_data):
    input_df = pd.DataFrame([input_data])
    
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    input_scaled = scaler.transform(input_df)
    
    prediction = knn.predict(input_scaled)
    
    return "Completed" if prediction[0] == 1 else "Cancelled"


sample_input = {
    'Age': 30,
    'Gender': 'Male',
    'Loyalty Member': 'Yes',
    'Product Type': 'Smartphone',
    'Rating': 4,
    'Payment Method': 'Credit Card',
    'Total Price': 2000,
    'Unit Price': 500,
    'Quantity': 4,
    'Shipping Type': 'Express'
}

print("\nSample Prediction:", predict_customer(sample_input))