# --------------------------
# 1) LIBRARY IMPORTS & COLUMN NAMES
# --------------------------
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting
from sklearn.model_selection import train_test_split  # to split dataset
from sklearn.preprocessing import StandardScaler  # to scale features
from sklearn.metrics import confusion_matrix, classification_report  # evaluation metrics
from collections import Counter  # to count most frequent elements

# Define column names (assuming Class is in the first column)
col_names = [
    "Class",
    "Alcohol",
    "Malic_acid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280_OD315_of_diluted_wines",
    "Proline"
]

# Read the Wine dataset from a CSV file named "wine.data"
df = pd.read_csv(r"C:\Users\Salih\Desktop\dersler\ml\hw1\wine.data", names=col_names)

# Print the first 5 rows to verify correct loading
print("First 5 rows of the dataset:")
print(df.head())

# --------------------------
# 2) PREPROCESSING & SPLITTING
# --------------------------
# Separate features (X) and class labels (y)
X = df.drop("Class", axis=1).values  # all columns except 'Class'
y = df["Class"].values  # the Class column

# Create a StandardScaler to scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit to X, then transform it

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,  # scaled features
    y,  # class labels
    test_size=0.2,  # 20% as test data
    random_state=42
)
print("\nFirst 5 rows of the scaled X_scaled:")
print(X_scaled[:5])


# --------------------------
# 3) K-NN IMPLEMENTATION
# --------------------------
def euclidean_distance(a, b):
    # Calculate Euclidean (L2) distance between vectors a and b
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    # Calculate Manhattan (L1) distance between vectors a and b
    return np.sum(np.abs(a - b))


def knn_predict(X_train, y_train, x_test, k, dist_func):
    # Predict the class for a single test sample x_test using k-NN
    distances = []
    for i, x_train_sample in enumerate(X_train):
        dist = dist_func(x_train_sample, x_test)  # compute distance
        distances.append((dist, y_train[i]))  # store (distance, label)
    distances.sort(key=lambda x: x[0])  # sort by distance, ascending
    k_nearest = distances[:k]  # take the k closest
    labels = [label for _, label in k_nearest]  # extract their labels
    # return the most frequent label among the k nearest neighbors
    return Counter(labels).most_common(1)[0][0]


def knn_accuracy(X_train, y_train, X_test, y_test, k, dist_func):
    # Calculate accuracy of k-NN on the entire test set
    correct = 0
    for i, x_test_sample in enumerate(X_test):
        prediction = knn_predict(X_train, y_train, x_test_sample, k, dist_func)
        if prediction == y_test[i]:
            correct += 1
    return correct / len(X_test)


# --------------------------
# 4) EXPERIMENT: DIFFERENT K & METRICS
# --------------------------
# Choose some k values
k_values = [1, 3, 5, 7, 9]

# Dictionary of distance functions to try
dist_funcs = {
    "Euclidean": euclidean_distance,
    "Manhattan": manhattan_distance
}

# Dictionary to store accuracy results for plotting
results = {}

# Loop over each distance metric
for dist_name, dist_func in dist_funcs.items():
    print(f"\nUsing {dist_name} distance:")
    acc_list = []
    # For each k in k_values, calculate accuracy
    for k_val in k_values:
        acc = knn_accuracy(X_train, y_train, X_test, y_test, k_val, dist_func)
        acc_list.append(acc)
        print(f"  k={k_val}, Accuracy={acc:.3f}")
    results[dist_name] = acc_list

# --------------------------
# 5) CONFUSION MATRIX & REPORT
# --------------------------
# Suppose we pick the best combo from the results (e.g., k=5, Manhattan)
final_k = 5
final_dist_func = manhattan_distance  # example choice

# Predict labels on the test set
y_pred = [knn_predict(X_train, y_train, x_test_sample, final_k, final_dist_func)
          for x_test_sample in X_test]

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Generate classification report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:\n", cr)

# --------------------------
# 6) PLOT ACCURACY VS. K
# --------------------------
for dist_name, acc_list in results.items():
    plt.plot(k_values, acc_list, marker='o', label=dist_name)

plt.title("Accuracy vs. K (Wine Dataset)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Here, we visualize the first two features of X_scaled to see class separation.

plt.figure()
unique_classes = np.unique(y)
for c in unique_classes:
    # plot only rows where the label == c
    plt.scatter(
        X_scaled[y == c, 0],  # first feature (scaled Alcohol)
        X_scaled[y == c, 1],  # second feature (scaled Malic_acid)
        label=f"Class {c}",
        alpha=0.7
    )
plt.title("Scatter Plot of First Two Scaled Features by Class")
plt.xlabel("Scaled Alcohol")
plt.ylabel("Scaled Malic_acid")
plt.legend()
plt.show()

