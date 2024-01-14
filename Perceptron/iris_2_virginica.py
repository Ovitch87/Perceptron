from Perceptron_algorithm import Perceptron_x, unit_step_function, sgn_function, accuracy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels
target_names = iris.target_names  # Species names

# Features for the 3D plot
feature_indices = [0, 1, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each species
for species in range(3):
    indices = (y == species)
    ax.scatter(X[indices, feature_indices[0]], X[indices, feature_indices[1]], X[indices, feature_indices[2]], label=target_names[species])

# Train a perceptron to separate Virginica from the others
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Update target labels to differentiate Virginica (class 2) from the others
y_train_virginica = np.where(y_train == 2, 1, -1)
y_test_virginica = np.where(y_test == 2, 1, -1)

# Instantiate and train the perceptron
perceptron = Perceptron_x(iterations=1000, activation_function=sgn_function, random_seed=42)
perceptron.fit(X_train[:, :3], y_train_virginica)

# Make predictions on the test set
predictions = perceptron.predict(X_test[:, :3])
accuracy = accuracy(y_test_virginica, predictions)
print(f'Accuracy: {accuracy}')

# Plot the decision boundary (plane)
xx, yy = np.meshgrid(np.linspace(X[:, feature_indices[0]].min(), X[:, feature_indices[0]].max(), 100),
                     np.linspace(X[:, feature_indices[1]].min(), X[:, feature_indices[1]].max(), 100))
zz = (-perceptron.intercept_ - perceptron.coef_[0] * xx - perceptron.coef_[1] * yy) / perceptron.coef_[2]
ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

# Labeling axes
ax.set_xlabel(iris.feature_names[feature_indices[0]])
ax.set_ylabel(iris.feature_names[feature_indices[1]])
ax.set_zlabel(iris.feature_names[feature_indices[2]])

# Adding a legend
ax.legend()

plt.title('3D Plot of Iris Dataset with Species Names in Legend and Virginica Separation Plane')
plt.show()

# Plot the number of errors over iterations
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_)
plt.xlabel("Iteration")
plt.ylabel("Number of Errors")
plt.title("Perceptron Training - Number of Errors over Iterations")
plt.show()