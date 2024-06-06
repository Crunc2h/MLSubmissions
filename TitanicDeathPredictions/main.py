from NNFramework.Dependencies.DataManipulation.data_structure import DataStructure
from MyFailedNetwork.NNFramework.Network.NetworkTypes.Neural import Neural
from MyFailedNetwork.NNFramework.Network.Layers.dense import Dense
from MyFailedNetwork.NNFramework.Dependencies.Functions.activation_functions import ActivationFuncs
from MyFailedNetwork.NNFramework.Dependencies.Functions.cost_functions import CostFuncs
from MyFailedNetwork.NNFramework.Dependencies.Diagnostics.diagnostics import Diagnostics
import matplotlib.pyplot as plt
import numpy as np
import pandas
with open("train.csv") as raw_data:
    csv_data = pandas.read_csv(raw_data)
EMPTY_STRING = ''
NAN = 'nan'
CABIN_SECTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'FG', 'T', 'unknown']
EMBARKATION = ['C', 'Q', 'S', 'unknown']
ECO_CLASSES = ['1', '2', '3', 'unknown']
SEX = ['male', 'female', 'unknown']
TITLES = ['Unknown',
          'Mr',
          'Mrs',
          'Miss',
          'Master',
          'Don',
          'Rev',
          'Dr',
          'Mme',
          'Sir',
          'Ms',
          'Major',
          'Lady',
          'Mlle',
          'Col',
          'Capt',
          'the Countess',
          'Jonkheer']
DEF_CABIN_NUMS_SIZE = 6
DEF_CABIN_SECS_SIZE = 10
DEF_EMBARK_SIZE = 4
DEF_TITLE_SIZE = 18
N_FEATURES = 9
ALPHA = 0.00001
EPOCHS = 10

def get_ticket_number(ticket_string):
    number = []
    for i in range(len(ticket_string)):
        if ticket_string[i].isnumeric():
            number.append(ticket_string[i])
    return EMPTY_STRING.join(number) if len(number) > 0 else 0

def get_cabin_numbers_and_sections(cabin_string):

    cabins_associated = 0
    cabin_secs_arr = np.zeros((DEF_CABIN_SECS_SIZE))
    cabin_nums_arr = np.zeros((DEF_CABIN_NUMS_SIZE))
    if str(cabin_string) == NAN:
        return cabin_secs_arr, cabin_nums_arr, cabins_associated
    else:
        cabin_strings = [c_str for c_str in cabin_string.split(' ') if c_str != EMPTY_STRING]
        cabin_section = []
        cabin_numbers = []
        for i in range(len(cabin_strings)):
            c_sec = []
            c_num = []
            for f in range(len(cabin_strings[i])):
                val = cabin_strings[i][f]
                if str(val).isnumeric() is False:
                    c_sec.append(val)
                elif str(val).isnumeric():
                    c_num.append(str(val))
            cabins_associated += 1
            if len(c_sec) != 0:
                cabin_section.append(EMPTY_STRING.join(c_sec))
            if len(c_num) != 0:
                cabin_numbers.append(int(EMPTY_STRING.join(c_num)))
            else:
                cabin_numbers.append(0)

        cabin_section = EMPTY_STRING.join(
            [cabin_section[i] for i in range(len(cabin_section)) if i == 0 or cabin_section[i - 1] != cabin_section[i]])
        if cabin_section in CABIN_SECTIONS:
            cabin_secs_arr[CABIN_SECTIONS.index(cabin_section)] = 1
        for i in range(len(cabin_numbers)):
            cabin_nums_arr[i] = cabin_numbers[i]

    return cabin_secs_arr, cabin_nums_arr, cabins_associated

# (cabin_section, cabin_numbers, num_cabins_associated)

cabin_related_features = [get_cabin_numbers_and_sections(row[1].values[-2]) for row in csv_data.iterrows()]
cabin_secs = [cab_feat [0] for cab_feat in cabin_related_features]
cabin_nums = [cab_feat [1] for cab_feat in cabin_related_features]
cabin_asc = [cab_feat [2] for cab_feat in cabin_related_features]
titles = [DataStructure.one_hot_code(name.split(', ')[1].split('.')[0], TITLES, DEF_TITLE_SIZE) for name in csv_data["Name"]]
embarkations = [np.zeros((DEF_EMBARK_SIZE)) if str(embark) == NAN else DataStructure.one_hot_code(embark, EMBARKATION, DEF_EMBARK_SIZE) for embark in csv_data["Embarked"]]
economic_classes = [DataStructure.one_hot_code(str(eco_class), ECO_CLASSES, len(ECO_CLASSES)) for eco_class in csv_data["Pclass"]]
genders = [DataStructure.one_hot_code(sex, SEX, len(SEX)) for sex in csv_data["Sex"]]
ages = [0 if str(age) == NAN else int(age) for age in csv_data["Age"]]
tickets = [0 if str(ticket) == NAN else int(get_ticket_number(str(ticket))) for ticket in csv_data["Ticket"].values]
fares = [value for value in csv_data["Fare"].values]
sibsp = [value for value in csv_data["SibSp"].values]
parch = [value for value in csv_data["Parch"].values]


def model_person(sex, c_sec, c_asc, title, embark, eco, age, sibsp, parch):
    packaged_vals = np.array([c_asc, age, sibsp, parch])
    stacked_arrs = np.hstack((sex, c_sec, title, embark, eco))
    return np.hstack((stacked_arrs, packaged_vals))

x_train_all = np.vstack([model_person(np.argmax(genders[i]),
                        np.argmax(cabin_secs[i]),
                        cabin_asc[i],
                        np.argmax(titles[i]),
                        np.argmax(embarkations[i]),
                        np.argmax(economic_classes[i]),
                        ages[i],
                        sibsp[i],
                        parch[i]) for i in range(len(tickets))])


y_train_all = np.array(csv_data["Survived"])
x_train_batches, y_train_batches = DataStructure.seperate_batches(x_train_all, y_train_all, 16)
network = Neural(layers=[Dense(n_features=N_FEATURES,
                               n_neurons=512,
                               activation_func=ActivationFuncs.relu,
                               derivative_func=ActivationFuncs.relu_derivative,
                               random_bias_init=True),
                         Dense(n_features=512,
                                n_neurons=256,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=256,
                                n_neurons=256,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=256,
                                n_neurons=256,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=256,
                                n_neurons=512,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=512,
                                n_neurons=128,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=128,
                                n_neurons=32,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=32,
                                n_neurons=8,
                                activation_func=ActivationFuncs.relu,
                                derivative_func=ActivationFuncs.relu_derivative,
                                random_bias_init=True),
                         Dense(n_features=8,
                                n_neurons=1,
                                activation_func=ActivationFuncs.sigmoid_activation,
                                derivative_func=ActivationFuncs.sigmoid_derivative,
                                random_bias_init=True), ],
                 x=x_train_batches,
                 y=y_train_batches,
                 cost_func=CostFuncs.mean_squared_error,
                 accuracy_func=Diagnostics.one_hot_bce_sigmoid_accuracy,
                 alpha=ALPHA,
                 epochs=EPOCHS)
train_predictions, train_costs, train_accuracies = network.compile(batching_mode=True, batch_size=16)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))
ax1.scatter(range(len(train_costs)), train_costs, c='r', marker='x')
ax2.scatter(range(len(train_accuracies)), train_accuracies, c='b', marker='x')
plt.show()



