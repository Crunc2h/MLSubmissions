from MyFailedNetwork.NNFramework.Dependencies.DataManipulation.data_manipulation import DataManipulation
from MyFailedNetwork.NNFramework.Dependencies.Functions.activation_functions import ActivationFuncs
from MyFailedNetwork.NNFramework.Dependencies.Functions.cost_functions import CostFuncs
from MyFailedNetwork.NNFramework.Dependencies.Diagnostics.diagnostics import Diagnostics
from MyFailedNetwork.NNFramework.Network.neural import Neural
from MyFailedNetwork.NNFramework.Network.Layers.dense import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas

with open("mnist_train.csv") as raw_data:
    csv_data = pandas.read_csv(raw_data)
    csv_data = np.array([csv_data]).reshape(60000, 785)
    training_labels = csv_data[:1000, 0].reshape(1, 1000)
    training_features = csv_data[:1000, 1:] / 255
    cv_labels = csv_data[2000:3000, 0].reshape(1, 1000)
    cv_features = csv_data[2000:3000, 1:] / 255
    test_labels = csv_data[1000:2000, 0].reshape(1, 1000)
    test_features = csv_data[1000:2000, 1:] / 255

N_CLASSES = 10
ALPHA = 0.0001
EPOCHS = 250
BATCH_SIZE = 1

y_train_ohc = [DataManipulation.one_hot_code(label_index=y_label,
                                             n_categories=N_CLASSES) for y_label in training_labels[0]]
y_cv_ohc = [DataManipulation.one_hot_code(label_index=y_label,
                                          n_categories=N_CLASSES) for y_label in cv_labels[0]]
y_test_ohc = [DataManipulation.one_hot_code(label_index=y_label,
                                            n_categories=N_CLASSES) for y_label in test_labels[0]]

y_train, y_cv, y_test = ([row for row in y_train_ohc],
                         [row for row in y_cv_ohc],
                         [row for row in y_test_ohc])

x_train, x_cv, x_test = ([row for row in training_features],
                         [row for row in cv_features],
                         [row for row in test_features])

train_batches = DataManipulation.create_randomized_data_batches(x_train, y_train, BATCH_SIZE)
cv_batches = DataManipulation.create_randomized_data_batches(x_cv, y_cv, BATCH_SIZE)
test_batches = DataManipulation.create_randomized_data_batches(x_test, y_test, BATCH_SIZE)

network = Neural(layers=[Dense(784, 16, ActivationFuncs.relu,
                               ActivationFuncs.relu_derivative),
                         Dense(16, 10, ActivationFuncs.softmax,
                               ActivationFuncs.softmax_derivative)],
                 cost_func=CostFuncs.categorical_cross_entropy,
                 accuracy_func=Diagnostics.one_hot_cce_softmax_accuracy)

training_cost_values, training_accuracy_values = network.compile(train_batches, ALPHA, EPOCHS)
cv_cost_values, cv_accuracy_values = network.compile(cv_batches, ALPHA, EPOCHS)
test_cost_values, test_accuracy_values = network.compile(test_batches, ALPHA, EPOCHS)

# GUI FOR PERFORMANCE AND DIAGNOSTICS

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(30, 5))

ax2.plot(range(EPOCHS), training_accuracy_values, c='r', label="Training Accuracy")
ax2.plot(range(EPOCHS), cv_accuracy_values, c='b', label="Cross Validation Accuracy")
ax2.plot(range(EPOCHS), test_accuracy_values, c='g', label="Test Accuracy")
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Iterations')
ax2.set_title('Accuracy Over Passes')

ax3.plot(range(EPOCHS), training_cost_values, c='r', label="J-training")
ax3.plot(range(EPOCHS), cv_cost_values, c='b', label="J-cv")
ax3.plot(range(EPOCHS), test_cost_values, c='g', label="J-test")
ax3.set_ylabel('J')
ax3.set_xlabel('Iterations')
ax3.set_title('J Across Passes')

ax2.legend()
ax3.legend()

plt.show()
