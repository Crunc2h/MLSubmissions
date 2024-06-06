from MyFailedNetwork.NNFramework.Network.Model.batch import Batch
import random
import numpy as np


class DataManipulation:
    @staticmethod
    def create_randomized_data_batches(x_set, y_set, batch_size):

        if len(x_set) != len(y_set) or len(x_set) < batch_size:
            raise AttributeError

        all_batches = []
        chosen_units_indexes = []
        m = len(x_set)
        n_batches = m // batch_size + 1

        for i in range(n_batches):

            batch_x = []
            batch_y = []

            if i == n_batches - 1:
                residue_indexes_count = m % batch_size
                batch_size = residue_indexes_count

            for f in range(batch_size):
                rand_choice = random.randint(0, m - 1)
                while rand_choice in chosen_units_indexes:
                    rand_choice = random.randint(0, m - 1)

                batch_x.append(np.array(x_set[f]))
                batch_y.append(np.array(y_set[f]))

                chosen_units_indexes.append(f)

            if batch_x:
                batch_x, batch_y = np.vstack(batch_x), np.vstack(batch_y)
                all_batches.append(Batch(batch_x, batch_y))

        return all_batches

    @staticmethod
    def one_hot_code(label_index, n_categories):
        one_hot_zeros = np.zeros(n_categories)
        one_hot_zeros[label_index] = 1
        return one_hot_zeros
