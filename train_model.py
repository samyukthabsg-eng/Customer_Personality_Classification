import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("dataset/marketing_campaign.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# 2. Separate features and target
X = data.drop("Customer_Segment", axis=1)
y = data["Customer_Segment"]


# 3. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 6. Train model
model.fit(X_train, y_train)


# 7. Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 8. Save model
joblib.dump(model, "model/customer_personality_model.pkl")

# 9. Save label encoder
joblib.dump(
    label_encoder,
    "model/label_encoder.pkl"
)

# 10. Save feature names
joblib.dump(
    list(X.columns),
    "model/feature_names.pkl"
)

print("\nModel saved successfully!")
print("Files created inside the model folder.")