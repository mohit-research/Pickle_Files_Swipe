import pickle
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
def unpickling(x):
        return pickle.load(open(x, 'rb'))


def plot_graph(x_before, x_after):
        x_before = unpickling(x_before)[0]
        x_after = unpickling(x_after)[0]

        print (x_before.shape, x_after.shape)
        pca = PCA(n_components=2)

        x_before = pca.fit_transform(x_before)
        x_after = pca.fit_transform(x_after)

        # x_before = StandardScaler().fit_transform(x_before)
        # x_after = StandardScaler().fit_transform(x_after)

        x_before = np.array(x_before)
        x_after = np.array(x_after)
        mini = min(np.amin(x_before), np.amin(x_after))
        maxi = max(np.amax(x_before), np.amax(x_after))
        x_before = (x_before - np.amin(x_before))/ (np.amax(x_before) - np.amin(x_before))
        x_after = (x_after - np.amin(x_after))/ (np.amax(x_after) - np.amin(x_after))

        aset = set([tuple(x) for x in x_before])
        bset = set([tuple(x) for x in x_after])

        x_diff = bset.difference(aset)

        x_diff = np.array(list(x_diff))
        print (x_diff.shape)
        plt.scatter(x_before[:,0], x_before[:,1], color = 'green', label = 'Legitimate')
        plt.scatter(x_diff[:,0], x_diff[:,1], color = 'red', label = 'Adversary', marker = 'x')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig('nonGAN6.pdf')
        plt.show()


plot_graph('X_Legit_NonGAN.pkl', 'X_Adver_NonGAN.pkl')
#plot_graph('X_Legit_GAN.pkl','X_Adver_GAN.pkl')
