"""Functions to help with data generation and organization.
"""
import numpy as np
import random as rand

def triplet_generator(X, Y, batch_size=32,
                        yield_none_target=True):
    """Generates triplet samples randomly for training a Triplet network

    # Arguments
        X: The data from which to sample. The first dimension should be num_samples
        Y: The class labels of shape (num_samples,)
        batch_size:
        yield_none_target: Whether to yield a tuple with the batch data and None,
        or alternatively, just the batch data.

    # Returns
        Yields batches of triplets of the form [batch_anchors, batch_positives, batch_negatives]
    """
    num_classes = Y.max() + 1

    # Organize data by class
    samples_by_class = [[] for i in range(num_classes)]
    for j in range(X.shape[0]):
        samples_by_class[Y[j]].append(X[j,...])

    while True:
        batch_list_a = []
        batch_list_p = []
        batch_list_n = []
        for batch_ind in range(batch_size):
            # choose two random classes
            class_inds = rand.sample(range(num_classes), 2)
            # choose two random samples from first class
            pos_inds = rand.sample(range(len(samples_by_class[class_inds[0]])), 2)
            # choose one sample from second class
            neg_ind = rand.sample(range(len(samples_by_class[class_inds[1]])), 1)[0]

            batch_list_a.append(samples_by_class[class_inds[0]][pos_inds[0]])
            batch_list_p.append(samples_by_class[class_inds[0]][pos_inds[1]])
            batch_list_n.append(samples_by_class[class_inds[1]][neg_ind])

        x_list = [np.stack(batch_list_a), np.stack(batch_list_p), np.stack(batch_list_n)]
        if yield_none_target:
            yield (x_list, None)
        else:
            yield x_list

def pair_generator(X, Y, batch_size):
    """Generates pair samples randomly for training Siamese network

    # Arguments
        X: The data from which to sample. The first dimension should be num_samples
        Y: The class labels of shape (num_samples,)
        batch_size:

    # Returns
        Yields batches of triplets of the form [batch_anchors, batch_positives, batch_negatives]
    """
    num_classes = Y.max() + 1

    # Organize data by class
    samples_by_class = [[] for i in range(num_classes)]
    for j in range(X.shape[0]):
        samples_by_class[Y[j]].append(X[j,...])

    while True:

        batch_list_1 = []
        batch_list_2 = []

        labels = np.random.randint(2, size=(batch_size,))
        for batch_ind in range(batch_size):
            
            if labels[batch_ind] == 1:
                class_ind = np.random.randint(num_classes)
                sample_inds = rand.sample(range(len(samples_by_class[class_ind])), 2)
                batch_list_1.append(samples_by_class[class_ind][sample_inds[0]])
                batch_list_2.append(samples_by_class[class_ind][sample_inds[1]])

            else:
                class_inds = rand.sample(range(10), 2)
                sample_inds = [np.random.randint(len(samples_by_class[class_inds[0]])),
                                np.random.randint(len(samples_by_class[class_inds[1]]))]
                batch_list_1.append(samples_by_class[class_inds[0]][sample_inds[0]])
                batch_list_2.append(samples_by_class[class_inds[1]][sample_inds[1]])

        yield ([np.stack(batch_list_1), np.stack(batch_list_2)], labels)