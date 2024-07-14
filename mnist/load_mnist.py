import torchvision.datasets as datasets
import torch

def load_mnist():
    mnist_trainset = datasets.MNIST(root='./data/mnist_tr', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data/mnist_ts', train=False, download=True, transform=None)
    
    n_pixels = 784
    # flatten the images
    mnist_trainset.data = mnist_trainset.data.view(-1, n_pixels)
    mnist_testset.data = mnist_testset.data.view(-1, n_pixels)
    # normalize the images
    mnist_trainset.data = mnist_trainset.data.type(torch.float32) / 255
    mnist_testset.data = mnist_testset.data.type(torch.float32) / 255

    return mnist_trainset.data, mnist_trainset.targets, mnist_testset.data, mnist_testset.targets
