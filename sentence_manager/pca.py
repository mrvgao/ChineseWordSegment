"""
Get PCA spaces.

Reference: [1]. http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def get_pca(data):
    data_dimension = np.array(data).shape[1]
    data = scale(data)
    pca = PCA(n_components=int(np.ceil(data_dimension/2)))
    pca.fit(data)
    return pca.components_


if __name__ == '__main__':

    X = np.random.rand(201, 200)

    eigenvector = get_pca(X)[0]

    print(type(eigenvector))
    assert eigenvector.shape == (200, ), eigenvector.shape
    print('test done!')
