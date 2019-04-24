"""Useful Callbacks for training Metric Learning models.
"""

from keras.models import Model
from keras.callbacks import Callback
import numpy as np
from types import GeneratorType
from kml_utils import recall_at_k, nmi

class RecallAtK(Callback):
    """Callback that computes the Recall@k metric for a given validation set at the end of each epoch.

    # Arguments:
        validation_data: Can be either a tuple of data and labels, or a generator that yields batches of tuples
        validation_steps: Only relevant if validation_data is a generator
        k: How many closest embeddings to consider when computing recall
        metric: The distance metric in the embedding space. Defaults to 'euclidean'.
        model_name: The name of the base network that maps input samples to their embeddings.\
            If not provided, the layer within the training network of type 'Model' will be selected.
        verbose: Whether to print the computed recall after each epoch.
    """

    def __init__(self, validation_data, validation_steps=1, k=1, metric='euclidean', model_name=None, verbose=False):
        super(RecallAtK, self).__init__()
        self.model_name = model_name
        self.k = k
        self.metric = metric
        # self.validation_data = validation_data
        # self.validation_steps = validation_steps
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        if 'recall_at_{}'.format(self.k) not in logs:
            logs['recall_at_{}'.format(self.k)] = []
        if epoch == 0:
            if self.model_name:
                self.model = self.model.get_layer(self.model_name)
            else:
                sub_models = [l for l in self.model.layers if isinstance(l, Model)]
                if len(sub_models) == 1:
                    self.model = sub_models[0]
        # if isinstance(self.validation_data, GeneratorType):
        #     val_embeddings = []
        #     labels = []
        #     for i in range(self.validation_steps):
        #         data, targets = self.validation_data.next()
        #         val_embeddings.append(self.model.predict(data))
        #         labels.extend(targets)
        #     val_embeddings = np.concatenate(val_embeddings, axis=0)
            
        # elif isinstance(self.validation_data, tuple) and len(self.validation_data) == 2:
        val_embeddings = self.model.predict(self.validation_data[0])
        labels = self.validation_data[1]

        # else:
        #     raise ValueError('validation_data must be either a generator object or a tuple (X,Y)')

        recall = recall_at_k(val_embeddings, labels, k=self.k, metric=self.metric)    
        logs['recall_at_{}'.format(self.k)].append(recall)

        if self.verbose:
            print '\nRecall@{} on validation: {}'.format(self.k, recall)


class NMI(Callback):
    """Callback that computes the Normalized Mutual Information score for the embeddings of a given validation set \
        at the end of each epoch.

    # Arguments:
        validation_data: Can be either a tuple of data and labels, or a generator that yields batches of tuples
        validation_steps: Only relevant if validation_data is a generator
        metric: The distance metric in the embedding space. Defaults to 'euclidean'.
        model_name: The name of the base network that maps input samples to their embeddings.
            If not provided, the layer within the training network of type 'Model' will be selected.
        verbose: Whether to print the computed NMI score after each epoch.
    """
    def __init__(self, validation_data, validation_steps=1, metric='euclidean', model_name=None, verbose=False):
        super(NMI, self).__init__()
        if model_name:
            self.model = self.model.get_layer(model_name)
        else:
            sub_models = [l for l in self.model.layers if isinstance(l, Model)]
            if len(sub_models) != 1:
                raise ValueError('Training network must contain exactly one sub-model')
            self.model = sub_models[0]

        self.metric = metric
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'nmi' not in logs:
            logs['nmi'] = []
        if isinstance(self.validation_data, GeneratorType):
            val_embeddings = []
            labels = []
            for i in range(self.validation_steps):
                data, targets = self.validation_data.next()
                val_embeddings.append(self.model.predict(data))
                labels.extend(targets)
            val_embeddings = np.concatenate(val_embeddings, axis=0)
            
        elif isinstance(self.validation_data, tuple) and len(self.validation_data == 2):
            val_embeddings = self.model.predict(self.validation_data[0])
            labels = self.validation_data[1]

        else:
            raise ValueError('validation_data must be either a generator object or a tuple (X,Y)')

        this_nmi = nmi(val_embeddings, labels, metric=self.metric)    
        logs['nmi'].append(this_nmi)

        if self.verbose:
            print '\nNMI on validation: {}'.format(this_nmi)