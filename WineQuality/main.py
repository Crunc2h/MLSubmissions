import matplotlib.pyplot as plt
import numpy as np
import pandas
with open("winequality-white.csv") as raw_data:
    csv_data = pandas.read_csv(raw_data)
    feature_arrs = []
    target_val_list = []
    for index, row in csv_data.iterrows():
        for column in csv_data.columns:
            row_values = [round(float(value), 4) for value in csv_data[column][index].split(';')]
            arr = np.array(row_values[:-1])
            target_val_list.append(row_values[-1])
            feature_arrs.append(arr)
    labels = np.array(target_val_list)
    features = np.vstack(feature_arrs)
MAX_FEATURE_VALUES = np.max(features, axis=0)
MIN_FEATURE_VALUES = np.min(features, axis=0)
AVG_FEATURE_VALUES = np.average(features, axis=0)


def scale_features(x_ndarray):
    m, n = x_ndarray.shape
    for i in range(n):
        feature_vals = x_ndarray[:,i]
        for f in range(m):
            feature_vals[f] = ((feature_vals[f] - AVG_FEATURE_VALUES[i])
                               / (MAX_FEATURE_VALUES[i] - MIN_FEATURE_VALUES[i]))
    return x_ndarray


def seperate_data(x, y, m):
    x_tr = x[0:int(m * 3 / 5), :]
    x_crossv = x[int(m * 3 / 5):int(m * 4 / 5), :]
    x_tst = x[int(m * 4 / 5):int(m * 5 / 5), :]
    y_tr = y[0:int(m * 3 / 5)]
    y_crossv = y[int(m * 3 / 5):int(m * 4 / 5)]
    y_tst = y[int(m * 4 / 5):int(m * 5 / 5)]
    return x_tr, y_tr, x_crossv, y_crossv, x_tst, y_tst


x_train, y_train, x_cv, y_cv, x_test, y_test = seperate_data(x=scale_features(x_ndarray=features),
                                                             y=labels,
                                                             m=features.shape[0])



