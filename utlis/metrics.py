from scipy import spatial


def cosine(v1, v2):
    eps = 1e-6
    try:
        distance = spatial.distance.cosine(v1, v2)
    except ValueError:
        distance = float('inf')

    return distance
