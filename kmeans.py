import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
import cv2
from scipy.misc import imsave


def add_image( fig, img, w, h, p, label):
    a = fig.add_subplot(w, h, p)
    a.axis('off')
    #cv2.imwrite("{}.png" .format(label), img)

    plt.imshow(img)
    a.set_title(label)


def kmeans_seg( img, num_clusters):
    # Convert input image into (num_samples, num_features) 
    # array to run kmeans clustering algorithm 
    X = img.reshape((-1, 1))  

    # Run kmeans on input data
    kmeans = KMeans(n_clusters=num_clusters,  n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    # Assign each value to the nearest centroid and 
    # reshape it to the original image shape
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed 



 # Leitura Imagem
img1 = imread('abie.jpg', as_gray = True)
img1 = (img1 * 255).round().astype(np.uint8)

cv2.imwrite("gray.jpg", img1)
limiar, imgLimiar = cv2.threshold( img1, 177, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("abie_limiar.png", imgLimiar)

res2 = kmeans_seg(imgLimiar, 2) / 255
cv2.waitKey(0)
cv2.destroyAllWindows()     

fig = plt.figure(figsize=(7,2), dpi=200)
add_image(fig, img1, 1, 4, 1, 'original')
add_image(fig, res2, 1, 4, 2, 'k=2')







