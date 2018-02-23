"""Functions that can be used as losses in model.compile()
"""

from keras import backend as K
import numpy as np
from keras.losses import categorical_crossentropy

class ContrastiveLoss(object):
    """Creates contrastive loss function usable in Model.compile()

    # Arguments
        repulsion_thresh: The threshold over which not to penalize the distance between samples\
        from different classes
        attraction_thresh: The threshold under which not to penalize the distance between samples\
        from the same class 
    """
    def __init__(self, repulsion_thresh, attraction_thresh=0.):
        self.attraction_thresh = attraction_thresh
        self.repulsion_thresh = repulsion_thresh
    
    def loss_function(self, labels, dists):
        loss = (1.-labels) * K.relu(self.repulsion_thresh - dists) + labels * K.relu(dists - self.attraction_thresh)
        return K.mean(loss)

