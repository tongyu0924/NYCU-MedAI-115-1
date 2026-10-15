import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def read_bci_data(data_dir=DATA_DIR):
    S4b_train = np.load(os.path.join(data_dir, 'S4b_train.npz'))
    X11b_train = np.load(os.path.join(data_dir, 'X11b_train.npz'))
    S4b_test = np.load(os.path.join(data_dir, 'S4b_test.npz'))
    X11b_test = np.load(os.path.join(data_dir, 'X11b_test.npz'))

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # labels 1/2 -> 0/1
    train_label = train_label - 1
    test_label = test_label - 1
    # [N, 750, 2] -> [N, 1, 2, 750]
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    # fill NaN samples with the mean value
    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label
