"""Useful custom layers for metric learning applications.
"""

from keras.layers import Layer
from keras import backend as K

def pairwise_dists(A,B, epsilon=1e-6):
    """Helper function to compute pairwise distances between rows in A and rows in B 
    """
    normsA = K.sum(A*A, axis=1)
    normsA = K.reshape(normsA, [-1, 1])
    normsB = K.sum(B*B, axis=1)
    normsB = K.reshape(normsB, [1,-1])
    dists = normsA - 2*K.tf.matmul(A, B, transpose_b=True) + normsB
    dists = K.sqrt(K.relu(dists)+epsilon)
    return dists

class TripletLoss(Layer):
    """Computes the triplet distance loss given the triplet embeddings.\
        The input to the layer is a list of tensors in the following order:\
        [anchor_embeddings, positive_embeddings, negative_embeddings].

        The output of the layer can be passed to Model.add_loss() as it is\
        intended ot be minimized directly without comparison to labels.

        # Arguments
        margin: The margin between inter-class distances and intra-class distances.
        epsilon: Small number to add before sqrt. Defaults to 1e-6

        # Input shapes
        list of 2D tensors with shapes: `[(batch_size, latent_dim), (batch_size, latent_dim), (batch_size, latent_dim)]`

        # Output shape
        1D Tensor with shape `(batch_size,)`
    """

    def __init__(self, margin, epsilon=1e-6, **kwargs):
        self.margin = margin
        self.epsilon = epsilon
        super(TripletLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TripletLoss, self).build(input_shape)

    def call(self, x):
        anchors = x[0]
        positives = x[1]
        negatives = x[2]
        pos_dists = K.sqrt(K.relu(K.sum(K.square(anchors-positives), axis=1))+self.epsilon)
        neg_dists = K.sqrt(K.relu(K.sum(K.square(anchors-negatives), axis=1))+self.epsilon)
        return K.relu(pos_dists - neg_dists + self.margin)


class PairDistances(Layer):
    """Computes the distances between corresponding samples in pairs of embeddings.

        # Arguments
        epsilon: Small number to add before sqrt. Defaults to 1e-6

        # Input shapes
        list of 2D tensors with shapes: `[(batch_size, latent_dim), (batch_size, latent_dim)]

        # Output shape
        1D Tensor with shape `(batch_size,)`

    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super(PairDistances, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PairDistances, self).build(input_shape)

    def call(self, x):
        dists = K.sqrt(K.relu(K.sum(K.square(x[0]-x[1]), axis=1))+self.epsilon)
        return K.expand_dims(dists, axis=-1)



class LiftedStructuredEmbedding(Layer):
    """Computes the Lifted Structured Feature Embedding as described in Song et al.\
        (https://arxiv.org/pdf/1511.06452.pdf) The output of this layer can be used\
        in Model.add_loss() as the embedding is to be minimized directly without comparison\
        to labels.
        
        This layer expects a structured batch of a specific number of classes and a specific\
        number of batches per class; all samples of the same class must be located consecutively\
        within the batch. These can be generated with data_utils.structured_batch_generator()

        # Arguments
        num_classes_per_batch: 
        num_samples_per_class: 
        margin: The margin to use when taking the logsumexp of the negative pair distances in the batch
        epsilon: Small number to add before sqrt. Defaults to 1e-6
    """
    def __init__(self, num_classes_per_batch, num_samples_per_class, margin, epsilon=1e-6, **kwargs):
        super(LiftedStructuredEmbedding, self).__init__(**kwargs)
        self.margin = margin
        self.epsilon = epsilon
        self.p = num_classes_per_batch
        self.k = num_samples_per_class

    def build(self, input_shape):
        super(LiftedStructuredEmbedding, self).build(input_shape)

    def call(self, x):
        # Construct the pairwise distance matrix
        D = pairwise_dists(x, x, epsilon=self.epsilon)
        J = []
        # We need to loop through all positive pairs. Since we know
        # the structure of the batch, this is not too difficult.
        for c in range(self.p): # Loop through classes
            for i in range(self.k):
                for j in range(i+1, self.k):
                    row_i = c*self.k + i
                    row_j = c*self.k + j
                    rows = K.gather(D, K.constant([row_i, row_j], dtype=K.tf.int32))
                    rows = K.concatenate([K.tf.slice(rows, begin=[0,0], size=[2,c*self.k]),
                            K.tf.slice(rows, begin=[0,(c+1)*self.k], size=[2,(self.p-c-1)*self.k])], axis=1)
                    rows = K.flatten(rows)
                    J.append(K.logsumexp(self.margin - rows) + D[row_i,row_j])
        
        J = K.stack(J)
        return K.mean(K.square(K.relu(J))) / 2.0

    def compute_output_shape(self, input_shape):
        return ()