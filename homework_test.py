'''
Homework 1
Rekha Saha
Panther Id: 002615612
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpg

# read image
load_image = mpg.imread('homework_1.jpg')
# Singular value decomposition
U, s, V = np.linalg.svd(load_image)
#component number list for image reconstruction
ncomps = [1,5,10,20,40]

for i in range(len(ncomps)):
    # reconstructed matrix
    recon_img = np.matrix(U[:, :ncomps[i]]) * np.diag(s[:ncomps[i]]) * np.matrix(V[:ncomps[i], :])
    # printing reconstructed image
    plt.imshow(recon_img,cmap='gray')
    plt.title('Reconstructed figure for k value = ' + str(ncomps[i]))
    plt.savefig("k_value_"+str(ncomps[i])+".png")
