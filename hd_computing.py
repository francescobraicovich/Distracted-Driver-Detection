import numpy as np

def create_level_hvs(num_levels, hd):
    level_hvs = np.zeros((num_levels, hd.dim))
    for i in range(num_levels):
        level_hvs[i] = hd.random_hv()
    return level_hvs

def create_position_hvs(num_positions, hd):
    position_hvs = np.zeros((num_positions, hd.dim))
    base_position_hv = hd.random_hv()
    position_hvs[0] = base_position_hv
    for pos in range(1, num_positions):
        position_hvs[pos] = hd.permute(position_hvs[pos - 1])
    return position_hvs

def encode_features(features, hd, num_levels, level_hvs, position_hvs):
    quantized_features = np.floor(features * (num_levels - 1)).astype(int)
    # Using -1 to index the length of the last axis, 
    # this makes it vecotrized and works for single and multiple features simultaneously
    encoded_features = np.zeros((features.shape[-1], level_hvs.shape[1]))
    encoded_features = level_hvs[quantized_features]
    encoded_features = hd.bind(encoded_features, position_hvs)
    encoded_hv = hd.superpose(encoded_features)
    return encoded_hv

class HDComputing():
    def __init__(self, dim):
        self.dim = dim

    def random_hv(self):
        return np.random.choice([1, -1], self.dim)
    
    def superpose(self, hvs):
        # Sum all the hypervectors corresponding to the features of the same vector
        # Using -2 to sum along the second last axis: (batch_size, num_features, dim)
        sum_hv = np.sum(hvs, axis=-2) 
        return np.sign(sum_hv)
    
    def bind(self, hv1, hv2):
        return hv1 * hv2
    
    def permute(self, hv, shifts=1):
        # Circular shift (permutation) to encode position
        return np.roll(hv, shifts)
    
    def vote(self, hvs, threshold):
        average_hv = np.average(hvs, axis=-2)
        mask1 = average_hv > threshold
        mask2 = average_hv < -threshold
        average_hv[mask1] = 1
        average_hv[mask2] = -1
        zero_mask = (~ mask1) & (~ mask2)
        average_hv[zero_mask] = 0
        return average_hv
    
    def hamming_similarity(self, hv1, hv2):
        return np.sum(hv1 == hv2) / self.dim
    
    def jaccard_similarity(self, hv1, hv2):
        intersection = (hv1 == 1) & (hv2 == 1)
        union = (hv1 == 1) | (hv2 == 1)
        return np.sum(intersection) / np.sum(union)
    
    def cosine_similarity(self, hv1, hv2):
        return np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))
    
    def average_similarity(self, hv1, hv2):
        hamming = self.hamming_similarity(hv1, hv2)
        jaccard = self.jaccard_similarity(hv1, hv2)
        cosine = self.cosine_similarity(hv1, hv2)
        return (hamming + jaccard + cosine) / 3

    
def create_level_hvs(num_levels, hd):
    level_hvs = np.zeros((num_levels, hd.dim))
    for i in range(num_levels):
        level_hvs[i] = hd.random_hv()
    return level_hvs

def create_position_hvs_old(num_positions, hd):
    position_hvs = np.zeros((num_positions, hd.dim))
    for pos in range(num_positions):
        position_hvs[pos] = hd.random_hv()
    return position_hvs

def create_position_hvs(num_positions, hd):
    position_hvs = np.zeros((num_positions, hd.dim))
    base_position_hv = hd.random_hv()
    position_hvs[0] = base_position_hv
    for pos in range(1, num_positions):
        position_hvs[pos] = hd.permute(position_hvs[pos - 1])
    return position_hvs

def encode_features(features, hd, num_levels, level_hvs, position_hvs):
    quantized_features = np.floor(features * (num_levels - 1)).astype(int)
    # Using -1 to index the length of the last axis, 
    # this makes it vecotrized and works for single and multiple features simultaneously
    encoded_features = np.zeros((features.shape[-1], level_hvs.shape[1]))
    encoded_features = level_hvs[quantized_features]
    encoded_features = hd.bind(encoded_features, position_hvs)
    encoded_hv = hd.superpose(encoded_features)
    return encoded_hv

def create_class_prototypes(encoded_hvs, labels, num_classes, hd, pecentile=0.5, voting_threshold=0.5):
    
    class_prototypes = np.zeros((num_classes, hd.dim))

    for i in range(num_classes):
        class_mask = labels == i
        class_hvs = encoded_hvs[class_mask]
        class_prototypes[i] = hd.vote(class_hvs, threshold=voting_threshold)

    similarities_to_prototypes = np.zeros(encoded_hvs.shape[0])

    for i in range(num_classes):
        class_mask = labels == i
        class_hvs = encoded_hvs[class_mask]
        prototype = class_prototypes[i]
        similarities = np.zeros(class_hvs.shape[0])
        
        for j in range(class_hvs.shape[0]):
            similarities[j] = hd.cosine_similarity(class_hvs[j], prototype)

        similarities_to_prototypes[class_mask] = similarities

    robust_class_prototypes = np.zeros((num_classes, hd.dim))
    for i in range(num_classes):
        class_mask = labels == i
        similarities = similarities_to_prototypes[class_mask]
        threshold = np.percentile(similarities, pecentile)
        really_close = similarities > threshold
        class_hvs = encoded_hvs[class_mask]
        really_close_hvs = class_hvs[really_close]
        robust_class_prototypes[i] = hd.superpose(really_close_hvs)

    return robust_class_prototypes