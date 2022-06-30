import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def unpickling(x):
    return pickle.load(open(x, 'rb'))


def plot_graph(x_before, x_after):
    x_before = unpickling(x_before)
    x_after = unpickling(x_after)
    pca = PCA(n_components=2)

    x_before = pca.fit_transform(x_before)
    x_after = pca.fit_transform(x_after)

    x_before = np.array(x_before)
    x_after = np.array(x_after)

    aset = set([tuple(x) for x in x_before])
    bset = set([tuple(x) for x in x_after])

    x_diff = bset.difference(aset)

    x_diff = np.array(list(x_diff))
    print (x_diff.shape)
    plt.scatter(x_before[:,0], x_before[:,1], color = 'green', label = 'Vanilla', markers = 'X')
    plt.scatter(x_diff[:,0], x_diff[:,1], color = 'blue', label = 'GAN')
    #plt.legend()
    plt.savefig('./nonGAN1.png')


    plot_graph('Plot/X_Legit_NonGAN_11.pkl', 'Plot/X_Adver_NonGAN_11.pkl')
    #plot_graph('X_train_phone_before_new_1.pkl','X_matrix_phone_after_new_1.pkl')
