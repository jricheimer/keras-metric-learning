"""Useful custom layers for metric learning applications.
"""

from keras.layers import Layer
from keras import backend as K

class TripletLoss(Layer):
    """Computes the triplet distance loss given the triplet embeddings.\
        The input to the layer is a list of tensors in the following order:\
        [anchor_embeddings, positive_embeddings, negative_embeddings].

        # Arguments:
        margin: The margin between inter-class distances and intra-class distances.
        epsilon: Small number to add before sqrt. Defaults to 1e-6

        # Input shapes
        list of 2D tensors with shapes: `[(batch_size, latent_dim), (batch_size, latent_dim), (batch_size, latent_dim)]`

        # Output shape
        1D Tensor with shape `(batch_size,)`
    """

    def __init__(self, margin, epsilon=1e-6 **kwargs):
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

    def __init__(self, epsilon=1e-6 **kwargs):
        self.epsilon = epsilon
        super(PairDistances, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PairDistances, self).build(input_shape)

    def call(self, x):
        dists = K.sqrt(K.relu(K.sum(K.square(x[0]-x[1]), axis=1))+self.epsilon)
        return dists

