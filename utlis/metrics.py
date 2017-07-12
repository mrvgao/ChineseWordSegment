from scipy import spatial
import numpy as np


def cosine(v1, v2):
    try:
        distance = spatial.distance.cosine(v1, v2)
    except ValueError:
        distance = float('inf')

    return distance
