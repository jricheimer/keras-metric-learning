"""Functions to help with data generation and organization.
"""
import numpy as np
import random as rand

def organize_by_class(x, y):
    """Organizes a dataset by class

    # Arguments
        x: numpy array with first dimension equal to num_samples
        y: list or ndarray of size (num_samples,) containing class ids for each corresponding member of x

    # Returns
        Dict mapping each class id to a numpy array of shape (num_samples_in_class,...)
    """
    class_ids = set(y)
    samples_by_class = {class_id: [] for class_id in class_ids}
    for j in range(x.shape[0]):
        samples_by_class[y[j]].append(x[j,...])
    for k in samples_by_class:
        samples_by_class[k] = np.stack(samples_by_class[k], axis=0)

    return samples_by_class
    

def triplet_generator(data, batch_size=32,
                        yield_none_target=True):
    """Generates triplet samples randomly for training a Triplet network

    # Arguments
        data: dict containing numpy arrays for each class, or h5py Group containing h5py Dataset for each class.
        batch_size:
        yield_none_target: Whether to yield a tuple with the batch data and None,
        or alternatively, just the batch data.

    # Returns
        Yields batches of triplets of the form [batch_anchors, batch_positives, batch_negatives]
    """
    class_ids = data.keys()

    while True:
        batch_list_a = []
        batch_list_p = []
        batch_list_n = []
        for _ in range(batch_size):
            # choose two random classes
            class_id_1, class_id_2 = rand.sample(class_ids, 2)
            # choose two random samples from first class
            pos_inds = rand.sample(range(data[class_id_1].shape[0]), 2)
            # choose one sample from second class
            neg_ind = rand.choice(range(data[class_id_2].shape[0]))

            batch_list_a.append(data[class_id_1][pos_inds[0],...])
            batch_list_p.append(data[class_id_1][pos_inds[1],...])
            batch_list_n.append(data[class_id_2][neg_ind,...])

        x_list = [np.stack(batch_list_a), np.stack(batch_list_p), np.stack(batch_list_n)]
        if yield_none_target:
            yield (x_list, None)
        else:
            yield x_list

def pair_generator(data, batch_size):
    """Generates pair samples randomly for training Siamese network

    # Arguments
        data: dict containing numpy arrays for each class, or h5py Group containing h5py Dataset for each class.
        batch_size:

    # Returns
        Yields batches of pairs of the form ([batch_1, batch_2], pairwise_labels)
    """
    class_ids = data.keys()

    while True:

        batch_list_1 = []
        batch_list_2 = []

        labels = np.random.randint(2, size=(batch_size,))
        for batch_ind in range(batch_size):
            
            if labels[batch_ind] == 1:
                class_id_1 = rand.choice(class_ids)
                sample_inds = rand.sample(range(data[class_id_1].shape[0]), 2)
                batch_list_1.append(data[class_id_1][sample_inds[0],...])
                batch_list_2.append(data[class_id_1][sample_inds[1],...])

            else:
                class_id_1, class_id_2 = rand.sample(class_ids, 2)
                sample_inds = [np.random.randint(data[class_id_1].shape[0]),
                                np.random.randint(data[class_id_2].shape[0])]
                batch_list_1.append(data[class_id_1][sample_inds[0],...])
                batch_list_2.append(data[class_id_2][sample_inds[1],...])

        yield ([np.stack(batch_list_1), np.stack(batch_list_2)], labels)


def structured_batch_generator(data, num_classes_per_batch, num_samples_per_class,
                                yield_none_target=True):
    """Generates structured batches for use with in-batch triplet-mining techniques,\
        such as Lifted Structured Feature Embedding and Batch-Hard Triplet Loss.

    # Arguments
        data: dict containing numpy arrays for each class, or h5py Group containing h5py Dataset for each class.
        num_classes_per_batch: Numer of randomly sampled classes to include in each batch
        num_samples_per_class:

    # Returns
        Yields batches of of batch size `num_classes_per_batch*num_samples_per_class`,\
        with the structure such that the batch is organized by class, i.e. if\
        num_samples_per_class=k, then the first k samples of the batch are of one class,
        the second k samples (k+1 - 2k) are from another class, etc.
    """
    class_ids = data.keys()
    num_classes = len(class_ids)
    
    if num_classes_per_batch > num_classes:
        raise ValueError('Structured batch cannot contain more classes than the \
                             dataset contains')

    while True:
        batch_list = []
        batch_class_ids = rand.sample(class_ids, num_classes_per_batch)
        for c_id in batch_class_ids:
            sample_inds = rand.sample(range(data[c_id].shape[0]), num_samples_per_class)
            batch_list.extend([data[c_id][s_ind,...] for s_ind in sample_inds])
        
        if yield_none_target:
            yield (np.stack(batch_list), None)
        else:
            yield np.stack(batch_list)
