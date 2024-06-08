
import numpy as np
import jax
import jax.numpy as jnp

def grahm_matrix(X):
    return jnp.dot(X, X.T)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.S = None
        self.V_r = None
        self.Z = None
        self.m = None
        
    def fit(self, X):
        X = X - jnp.mean(X, axis=0)
        self.m = X.shape[0]
        U, S, V = jnp.linalg.svd(X, full_matrices=False)
        self.S = S
        self.V_r = V[:self.n_components, :]
        self.Z = jnp.dot(X, self.V_r.T)
        
    def transform(self, X):
        return jnp.dot(X, self.V_r.T)
    
    def get_eigenvectors(self):
        return self.V_r
    
    def get_eigenvalues(self):
        return self.S ** 2 / (self.m - 1)
    
    def reconstruct(self, Z):
        return jnp.dot(Z, self.V_r)


def kernel_PCA(X, n_components, kernel):
    m = X.shape[0]
    K = kernel(X, X)
    K = K - jnp.mean(K, axis=0)
    K = K - jnp.mean(K, axis=1)
    U, S, V = jnp.linalg.svd(K)
    V_r = V[:, :n_components]
    Z = jnp.dot(K, V_r)
    return Z, V_r
    

def test_PCA():
    X = np.random.randn(10, 2)
    X[:, 1] = 2 * X[:, 0] + 1 + 1* np.random.randn(10)
    
    print(X)
    
    Z, V_r = PCA(X, 1)
    print(Z)
    # Create a 1x2 subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot X on the first subplot
    ax[0].scatter(X[:, 0], X[:, 1])
    for i, txt in enumerate(range(len(X))):
        ax[0].annotate(txt, (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
    ax[0].set_title('Scatter plot of X')
    ax[0].set_xlabel('X[:, 0]')
    ax[0].set_ylabel('X[:, 1]')

    # Plot Z on the second subplot
    # Z is 10x1, so we can just plot it against the index
    ax[1].scatter(Z[:, 0], np.zeros_like(Z))
    for i, txt in enumerate(range(len(Z))):
        ax[1].annotate(txt, (Z[i, 0], 0), textcoords="offset points", xytext=(0, 10), ha='center')
    ax[1].set_title('Scatter plot of Z')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Z')

    # plot the eigenvector
    ax[0].plot([0, V_r[0, 0]], [0, V_r[1, 0]], 'r-', lw=2)

    # Display the plot
    plt.tight_layout()
    plt.show()


def test_kernel_PCA():
        
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import make_circles
    # from sklearn.decomposition import KernelPCA

    X, _ = make_circles(n_samples=100, factor=0.3, noise=0.01)
    
    def rbf_kernel(X, Y, gamma=15):
        X = jnp.array(X)
        Y = jnp.array(Y)
        return jnp.exp(-gamma * jnp.linalg.norm(X[:, None] - Y[None], axis=-1))
    
    # Z, V_r = kernel_PCA(X, 3, rbf_kernel)
    Z, V_r = PCA(X, 3)
    
    fig = plt.figure(figsize=(12, 6))

    # Plot X on the first subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1])
    for i, txt in enumerate(range(len(X))):
        ax1.annotate(txt, (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
    ax1.set_title('Scatter plot of X')
    ax1.set_xlabel('X[:, 0]')
    ax1.set_ylabel('X[:, 1]')

    # Plot Z on the second subplot as a 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(Z[:, 0], Z[:, 1], Z[:, 2])
    for i, txt in enumerate(range(len(Z))):
        ax2.text(Z[i, 0], Z[i, 1], Z[i, 2], txt, size=10, zorder=1, color='k')
    ax2.set_title('3D scatter plot of Z')
    ax2.set_xlabel('Z[:, 0]')
    ax2.set_ylabel('Z[:, 1]')
    ax2.set_zlabel('Z[:, 2]')

    # Display the plot
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt    

    # test_PCA()

    # test_kernel_PCA()