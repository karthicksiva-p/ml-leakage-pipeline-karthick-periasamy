# =========================
# Task 1 — Data Leakage Example
# =========================

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# ❌ WRONG APPROACH (Data Leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ Fitting on entire dataset

# Split after scaling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# =========================
# Task 2 — Pipeline + Cross Validation
# =========================

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Correct split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Pipeline (no leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())

# =========================
# Task 3 — Decision Tree Depth Experiment
# =========================
from sklearn.tree import DecisionTreeClassifier

depths = [1, 5, 20]

results = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    results.append((depth, train_acc, test_acc))

# Display results
print("Depth | Train Accuracy | Test Accuracy")
for r in results:
    print(f"{r[0]}     | {r[1]:.4f}         | {r[2]:.4f}")
