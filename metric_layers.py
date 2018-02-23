"""Useful custom layers for metric learning applications.
"""

from keras.layers import Layer, Dense
from keras import backend as K
from keras import losses
from keras import initializers
import numpy as np

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

def remove_diag(A):
    """Helper function to remove diagonal elements from a matrix A 
    """
    num_rows = K.shape(A)[0]
    new_A = K.tf.boolean_mask(A, K.tf.logical_not(K.cast(K.eye(num_rows), K.tf.bool)))
    new_A = K.reshape(new_A, [num_rows, num_rows-1])
    return new_A

class TripletLoss(Layer):
    """Computes the triplet distance loss given the triplet embeddings.\
        The input to the layer is a list of tensors in the following order:\
        `[anchor_embeddings, positive_embeddings, negative_embeddings]`.\
        Note that this is the naive version of the triplet loss; no in-batch\
        mining is done for "hard" or "semi-hard" triplets. 

        The output of the layer can be passed to Model.add_loss() as it is\
        intended to be minimized directly without comparison to labels.

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
        return K.mean(K.relu(pos_dists - neg_dists + self.margin))


class PairDistances(Layer):
    """Computes the distances between corresponding samples in pairs of embeddings.
        The output of this layer can be used in a loss function, given pairwise labels\
        for whether or not the pair is similar.

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
        number of samples per class; all samples of the same class must be located consecutively\
        within the batch. These can be generated with `data_utils.structured_batch_generator()`

        # Arguments
        num_classes_per_batch: 
        num_samples_per_class: 
        margin: The margin to use when taking the logsumexp of the negative pair distances in the batch
        epsilon: Small number to add before sqrt. Defaults to 1e-6
        
        # Input shape
        2D tensor with shape: `(batch_size, latent_dim)` (where `batch_size` is `num_classes_per_batch`*`num_samples_per_class`)

        # Output shape
        Scalar Tensor 


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


class BatchHardTripletLoss(Layer):
    """Computes the Batch-Hard Triplet Loss as described in Hermans et al.\
        (https://arxiv.org/pdf/1703.07737.pdf) The output of this layer can be used\
        in Model.add_loss() as it is to be minimized directly without comparison\
        to labels.
        
        This layer expects a structured batch of a specific number of classes and a specific\
        number of samples per class; all samples of the same class must be located consecutively\
        within the batch. These can be generated with `data_utils.structured_batch_generator()`

        # Arguments
        num_classes_per_batch: 
        num_samples_per_class: 
        margin: The margin to use when taking the logsumexp of the negative pair distances in the batch
        epsilon: Small number to add before sqrt. Defaults to 1e-6
        use_softplus: Whether to use the softplus activation on the distance differences, as recommended in the paper.\
            If false, ReLU is used. Defualts to `True`.
        
        # Input shape
        2D tensor with shape: `(batch_size, latent_dim)` (where `batch_size` is `num_classes_per_batch`*`num_samples_per_class`)

        # Output shape
        1D Tensor with shape: `(batch_size,)`
    """

    def __init__(self, num_classes_per_batch, num_samples_per_class, margin,
                    epsilon=1e-6, use_softplus=True, **kwargs):
        super(BatchHardTripletLoss, self).__init__(**kwargs)
        self.margin = margin
        self.epsilon = epsilon
        self.use_softplus = use_softplus
        self.p = num_classes_per_batch
        self.k = num_samples_per_class

    def build(self, input_shape):
        super(BatchHardTripletLoss, self).build(input_shape)

    def call(self, x):
        # Construct the pairwise distance matrix
        D = pairwise_dists(x, x, epsilon=self.epsilon)
        # get the max intra-class distance for each sample
        max_pos = [K.max(K.tf.slice(D, begin=[i*self.k, i*self.k], size=[self.k, self.k]), axis=1) for i in range(self.p)]
        max_pos = K.concatenate(max_pos, axis=0)
        # get the min inter-class distance for each sample
        min_neg = []
        for i in range(self.p):
            left = K.tf.slice(D, begin=[i*self.k, 0], size=[self.k, i*self.k])
            right = K.tf.slice(D, begin=[i*self.k, (i+1)*self.k], size=[self.k, (self.p-i-1)*self.k])
            min_neg.append(K.min(K.concatenate([left, right], axis=1), axis=1))
        min_neg = K.concatenate(min_neg, axis=0)
        
        if self.use_softplus:
            return K.mean(K.softplus(self.margin + max_pos - min_neg))
        else:
            return K.mean(K.relu(self.margin + max_pos - min_neg))

    def compute_output_shape(self, input_shape):
        return ()


class NPairsEmbedding(Layer):
    """Computes the N-pair Embedding Loss as described in K. Sohn.\
        (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)\
        The output of this layer can be used\
        in Model.add_loss() as the embedding is to be minimized directly without comparison\
        to labels.
        
        This layer expects a structured batch of a specific number of classes and a two\
        samples per class; all samples of the same class must be located consecutively\
        within the batch. These can be generated with\
        `data_utils.structured_batch_generator(num_samples_per_class=2)`

        # Arguments
        num_classes_per_batch: 
        margin: The margin to use when taking the logsumexp of the negative pair distances in the batch
        epsilon: Small number to add before sqrt. Defaults to 1e-6
        reg_coefficient: Coefficient for regularization of the feature embedding norms.\
            Defaults to 1.0
        
        # Input shape
        2D tensor with shape: `(batch_size, latent_dim)` (where `batch_size` is `2*num_classes_per_batch`)

        # Output shape
        Scalar Tensor 

    """
    def __init__(self, num_classes_per_batch, margin, reg_coefficient=0.002, **kwargs):
        super(NPairsEmbedding, self).__init__(**kwargs)
        self.margin = margin
        self.p = num_classes_per_batch
        self.reg_coeff = reg_coefficient
        

    def build(self, input_shape):
        super(NPairsEmbedding, self).build(input_shape)
        labels = np.zeros((2*self.p, 2*self.p-1))
        for i in range(self.p):
            labels[2*i:2*(i+1), 2*i] = 1.
        self.labels = self.add_weight(name='labels', shape=labels.shape,
                                        initializer=initializers.Constant(value=labels),
                                        trainable=False)

    def call(self, x):
        embedding_norms = K.tf.norm(x, axis=1)

        # Construct the inner-product matrix
        F = K.tf.matmul(x, x, transpose_b=True)
        # Remove diagonal elements
        F = K.tf.boolean_mask(F, K.tf.logical_not(K.cast(K.eye(2*self.p), K.tf.bool)))
        F =  K.reshape(F, [2*self.p, 2*self.p-1])

        return K.mean(K.categorical_crossentropy(target=self.labels, output=F, from_logits=True))
                    + self.reg_coeff * K.mean(embedding_norms)

    def compute_output_shape(self, input_shape):
        return ()


class ClusterLoss(Dense):
    """This layer (1) returns a classification output obtained by applying a fully-connected
        layer with softmax activation on the inputs when called; and (2) supplies a loss
        function to use which incorporates the "cluster loss" (or "center loss") on the embeddings
        proposed by Wen et al. (http://ydwen.github.io/papers/WenECCV16.pdf)

        [Admittedly, this is a hacky way of achieving this, necessary only due to restrictions
        imposed by Keras; namely, that only the loss function has access to the labels, but that the
        loss function has no access to intermediate features of the model, and additionally that
        its inputs must be of identical shape.]

        # Arguments
            num_classes: How many classes in the training dataset.
            center_loss_coeff: How much weight is given to the center loss relative
                to the cross-entropy loss.
            center_update_rate: The rate at which to update the class cluster centers.

        # Input shape
            2D Tensor of shape: `(batch_size, latent_dim)`

        # Output shape
            2D Tensor of shape: `(batch_size, num_classes)`
    """
    def __init__(self, num_classes, center_loss_coeff=0.1, center_update_rate=0.5, **kwargs):
        super(ClusterLoss, self).__init__(num_classes, activation='softmax', **kwargs)
        self.num_classes = num_classes
        self.center_loss_coeff = center_loss_coeff
        self.center_update_rate = center_update_rate
        self.embeddings = None

    def build(self, input_shape):
        super(ClusterLoss, self).build(input_shape)
        self.centers = self.add_weight(shape=(self.num_classes, input_shape[-1]),
                                        name='centers',
                                        trainable=False,
                                        initializer='uniform')

    def call(self, x):
        self.embeddings = x
        return super(ClusterLoss, self).call(x)

    def loss_function(self, y_true, y_pred):
        xent_loss = losses.categorical_crossentropy(y_true, y_pred)

        # Compute the distances from the current class-centers to add to loss
        class_ids = K.tf.argmax(y_true, axis=1, output_type=K.tf.int32)
        these_class_centers = K.gather( self.centers, class_ids )
        center_dist = K.sum(K.square(self.embeddings-these_class_centers), axis=1)
        center_loss = 0.5 * K.mean(center_dist)

        # How many samples per class in this batch:
        class_nums = K.sum(y_true, axis=0)
    
        # compute the differences for the class centers update
        diffs_arr = []
        for i in range(self.num_classes):
            diff = K.cast(K.expand_dims(K.equal(class_ids, i),axis=-1), K.tf.float32) * (these_class_centers-self.embeddings)
            diff = K.sum(diff, axis=0, keepdims=True)
            diff = diff / (K.gather(class_nums, i)+1)
            diffs_arr.append(diff)
        diffs = K.concatenate(diffs_arr, axis=0)
        
        # Update class centers:
        self.add_update(K.update_sub(self.centers, self.center_update_rate * diffs), inputs=self.embeddings)        

        return xent_loss + self.center_loss_coeff * center_loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(self.num_classes))