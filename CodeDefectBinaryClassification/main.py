import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dense
from numpy import exp, array

M_TRAIN = 90000
M_CV = 30000
M_TEST = 30000
EPOCHS = 15
BATCH_SIZE = 16
INITIAL_ALPHA = 0.00000002
L1_MOD = 0.003

with open("OVERSAMPLED2.4.csv") as raw_data:
    data = pd.read_csv(raw_data)
    print(data.shape)
    data_x = data.drop("defects", axis=1)
    x_train, x_cv, x_test, x_final_test = (array(data_x.iloc[0:M_TRAIN, :]),
                                           array(data_x.iloc[M_TRAIN:M_TRAIN + M_CV, :]),
                                           array(data_x.iloc[M_TRAIN + M_CV: M_TRAIN + M_CV + M_TEST]),
                                           array(data_x.iloc[M_TRAIN + M_CV + M_TEST:, :]))
    data_y = data.defects
    y_train, y_cv, y_test, y_final_test = (array(data_y.iloc[0:M_TRAIN]),
                                           array(data_y.iloc[M_TRAIN:M_TRAIN + M_CV]),
                                           array(data_y.iloc[M_TRAIN + M_CV:M_TRAIN + M_CV + M_TEST]),
                                           array(data_y.iloc[M_TRAIN + M_CV + M_TEST:]))

model = tf.keras.models.Sequential([Dense(units=32, activation="relu", kernel_regularizer=l1(L1_MOD)),
                                    Dense(units=16, activation="relu", kernel_regularizer=l1(L1_MOD)),
                                    Dense(units=16, activation="relu", kernel_regularizer=l1(L1_MOD)),
                                    Dense(units=1, activation="linear", kernel_regularizer=l1(L1_MOD))])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.binary_accuracy,
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.Precision(),
                       ])

train_hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
cv_hist = model.fit(x_cv, y_cv, epochs=EPOCHS, batch_size=BATCH_SIZE)
test_hist = model.fit(x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

train_loss, train_acc, train_prec, train_rec = (train_hist.history["loss"],
                                                train_hist.history["binary_accuracy"],
                                                train_hist.history["precision"],
                                                train_hist.history["recall"])

cv_loss, cv_acc, cv_prec, cv_rec = (cv_hist.history["loss"],
                                    cv_hist.history["binary_accuracy"],
                                    cv_hist.history["precision"],
                                    cv_hist.history["recall"])

test_loss, test_acc, test_prec, test_rec = (test_hist.history["loss"],
                                            test_hist.history["binary_accuracy"],
                                            test_hist.history["precision"],
                                            test_hist.history["recall"])


def softmax(x):
    return 1 / 1 - exp(-x)


predictions = [1 if prediction > .5 else 0 for prediction in softmax(model.predict(x_final_test))]
accuracy = len([True for i in range(len(y_final_test)) if predictions[i] == y_final_test[i]]) / len(y_final_test)
print(accuracy * 100)

fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, figsize=(25,25))

ax1.set_title("Loss")
ax1.plot(train_loss, c='r', label="Train")
ax1.plot(cv_loss, c='g', label="Cv")
ax1.plot(test_loss, c='b', label="Test")

ax2.set_title("Accuracy")
ax2.plot(train_acc, c='r', label="Train")
ax2.plot(cv_acc, c='g', label="Cv")
ax2.plot(test_acc, c='b', label="Test")

ax3.set_title("Precision")
ax3.plot(train_prec, c='r', label="Train")
ax3.plot(cv_prec, c='g', label="Cv")
ax3.plot(test_prec, c='b', label="Test")

ax4.set_title("Recall")
ax4.plot(train_rec, c='r', label="Train")
ax4.plot(cv_rec, c='g', label="Cv")
ax4.plot(test_rec, c='b', label="Test")

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()


