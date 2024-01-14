# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets

# # Load the Iris dataset
# iris = datasets.load_iris()
# X = iris.data  # The feature matrix
# y = iris.target  # The labels
# target_names = iris.target_names  # Species names

# # Choose three features for the 3D plot
# feature_names = iris.feature_names
# feature_indices = [0, 1, 2]  # Change these indices to visualize different features

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot for each species
# for species in range(3):
#     indices = (y == species)
#     ax.scatter(X[indices, feature_indices[0]], X[indices, feature_indices[1]], X[indices, feature_indices[2]], label=target_names[species])

# # Labeling axes
# ax.set_xlabel(feature_names[feature_indices[0]])
# ax.set_ylabel(feature_names[feature_indices[1]])
# ax.set_zlabel(feature_names[feature_indices[2]])

# # Adding a legend
# ax.legend()

# plt.title('3D Plot of Iris Dataset with Species Names in Legend')
# plt.show()

#------------------------------------------------------xx------------------------------------------------------

# from Perceptron_algorithm import Perceptron_x, unit_step_function, sgn_function, accuracy
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# from sklearn import datasets

# # load Iris dataset
# iris = datasets.load_iris()
# X = iris.data  # features
# y = iris.target  # labels
# target_names = iris.target_names  # species names

# # features for the 3D plot
# feature_indices = [0, 1, 2, 3]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot for each species
# for species in range(3):
#     indices = (y == species)
#     ax.scatter(X[indices, feature_indices[0]], X[indices, feature_indices[1]], X[indices, feature_indices[2]], label=target_names[species])

# # Train a perceptron to separate Setosa from the others
# setosa_indices = (y == 0)
# other_indices = ~setosa_indices
# X_setosa = X[setosa_indices][:, feature_indices]
# X_other = X[other_indices][:, feature_indices]

# perceptron = Perceptron_x(random_seed=42)
# perceptron.fit(np.vstack((X_setosa, X_other)), np.hstack((np.ones(X_setosa.shape[0]), -np.ones(X_other.shape[0]))))

# # Plot the decision boundary (plane)
# xx, yy = np.meshgrid(np.linspace(X[:, feature_indices[0]].min(), X[:, feature_indices[0]].max(), 100),
#                      np.linspace(X[:, feature_indices[1]].min(), X[:, feature_indices[1]].max(), 100))
# zz = (-perceptron.intercept_ - perceptron.coef_[0] * xx - perceptron.coef_[1] * yy) / perceptron.coef_[2]
# ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

# # Labeling axes
# ax.set_xlabel(iris.feature_names[feature_indices[0]])
# ax.set_ylabel(iris.feature_names[feature_indices[1]])
# ax.set_zlabel(iris.feature_names[feature_indices[2]])

# # Adding a legend
# ax.legend()

# plt.title('3D Plot of Iris Dataset with Species Names in Legend and Setosa Separation Plane')
# plt.show()

#------------------------------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn import datasets

# # Load the Iris dataset
# iris = datasets.load_iris()
# X = iris.data  # The feature matrix
# y = iris.target  # The labels
# target_names = iris.target_names  # Species names

# # Choose three features for the 3D plot
# feature_indices = [0, 1, 2]  # Change these indices to visualize different features

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot for each species
# for species in range(3):
#     indices = (y == species)
#     ax.scatter(X[indices, feature_indices[0]], X[indices, feature_indices[1]], X[indices, feature_indices[2]], label=target_names[species])

# # Train a perceptron to separate Setosa from the others
# setosa_indices = (y == 0)
# other_indices = ~setosa_indices
# X_setosa = X[setosa_indices][:, feature_indices]
# X_other = X[other_indices][:, feature_indices]

# perceptron = Perceptron(random_state=42)
# perceptron.fit(np.vstack((X_setosa, X_other)), np.hstack((np.ones(X_setosa.shape[0]), -np.ones(X_other.shape[0]))))

# # Plot the decision boundary (plane)
# xx, yy = np.meshgrid(np.linspace(X[:, feature_indices[0]].min(), X[:, feature_indices[0]].max(), 100),
#                      np.linspace(X[:, feature_indices[1]].min(), X[:, feature_indices[1]].max(), 100))
# zz = (-perceptron.intercept_ - perceptron.coef_[0][0] * xx - perceptron.coef_[0][1] * yy) / perceptron.coef_[0][2]
# ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# # Labeling axes
# ax.set_xlabel(iris.feature_names[feature_indices[0]])
# ax.set_ylabel(iris.feature_names[feature_indices[1]])
# ax.set_zlabel(iris.feature_names[feature_indices[2]])

# # Adding a legend
# ax.legend()

# plt.title('3D Plot of Iris Dataset with Species Names in Legend and Setosa Separation Plane')
# plt.show()

#------------------------------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn import datasets

# # Load the Iris dataset
# iris = datasets.load_iris()
# X = iris.data  # The feature matrix
# y = iris.target  # The labels

# # Choose three features for the 3D plot
# feature_indices = [0, 1, 2]  # Change these indices to visualize different features

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Train a perceptron for each pair of species (OvA strategy)
# for species in range(3):
#     binary_labels = (y == species).astype(int)
#     perceptron = Perceptron(random_state=42)
#     perceptron.fit(X[:, feature_indices], binary_labels)

#     # Plot the data points
#     ax.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], X[:, feature_indices[2]], c=y, cmap='viridis', edgecolor='k')

#     # Plot the decision boundary (plane)
#     xx, yy = np.meshgrid(np.linspace(X[:, feature_indices[0]].min(), X[:, feature_indices[0]].max(), 100),
#                          np.linspace(X[:, feature_indices[1]].min(), X[:, feature_indices[1]].max(), 100))
#     zz = (-perceptron.intercept_ - perceptron.coef_[0][0] * xx - perceptron.coef_[0][1] * yy) / perceptron.coef_[0][2]
#     ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# # Labeling axes
# ax.set_xlabel(iris.feature_names[feature_indices[0]])
# ax.set_ylabel(iris.feature_names[feature_indices[1]])
# ax.set_zlabel(iris.feature_names[feature_indices[2]])

# plt.title('3D Plot with Decision Boundaries for Each Pair of Species')
# plt.show()

#------------------------------------------------------------------------------------------------------------
from Perceptron_algorithm import Perceptron_x, unit_step_function, sgn_function, accuracy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets

# load Iris dataset
iris = datasets.load_iris()
X = iris.data  # features
y = iris.target  # labels
target_names = iris.target_names  # species names

# features for the 3D plot
feature_indices = [0, 1, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each species
for species in range(3):
    indices = (y == species)
    ax.scatter(X[indices, feature_indices[0]], X[indices, feature_indices[1]], X[indices, feature_indices[2]], label=target_names[species])

# Train a perceptron to separate Setosa from the others
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = np.where(y_train == 0, 1, -1)
y_test = np.where(y_test == 0, 1, -1)

perceptron = Perceptron_x(iterations=10, activation_function=sgn_function, random_seed=42)
perceptron.fit(X_train[:, :3], y_train)

predictions = perceptron.predict(X_test[:, :3])
print(predictions)
accuracy = accuracy(y_test, predictions)
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

plt.title('3D Plot of Iris Dataset with Species Names in Legend and Setosa Separation Plane')
plt.show()

print(perceptron.errors_)

plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_)
plt.xlabel("Iteration")
plt.ylabel("Number of Errors")

plt.show()   