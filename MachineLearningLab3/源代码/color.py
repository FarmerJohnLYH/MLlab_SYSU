import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# Load the photo and convert the color order
data = cv.imread("my.jpg")
# data = cv.imread("ladybug.png")
b, g, r = cv.split(data)
data = cv.merge((r, g, b))

# Convert to float and normalize for better visualization with plt.imshow (range [0-1])
data = np.array(data, dtype=np.float64) / 255

# Load the image and convert it to a 2D numpy array
w, h, d = original_shape = tuple(data.shape)
assert d == 3
image_array = np.reshape(data, (w * h, d))


def Get_processed_image(image_array, n_colors=10):
    # Train the k-means model
    kmeans = KMeans(n_init=1, n_clusters=n_colors, random_state=0)
    kmeans.fit(image_array)
    # Get the labels for all points
    kmeans_labels = kmeans.predict(image_array)
    return kmeans.cluster_centers_[kmeans_labels].reshape(w, h, -1)


plt.figure(num='my', figsize=(12, 9))

# Show the original image
plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("Original image", fontsize=10)
plt.imshow(data, aspect='auto')

n_colors = 32
plt.subplot(2, 3, 2)
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_colors), fontsize=10)
plt.imshow(Get_processed_image(image_array, n_colors), aspect='auto')

n_colors = 16
plt.subplot(2, 3, 3)
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_colors), fontsize=10)
plt.imshow(Get_processed_image(image_array, n_colors), aspect='auto')

n_colors = 8
plt.subplot(2, 3, 4)
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_colors), fontsize=10)
plt.imshow(Get_processed_image(image_array, n_colors), aspect='auto')

n_colors = 4
plt.subplot(2, 3, 5)
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_colors), fontsize=10)
plt.imshow(Get_processed_image(image_array, n_colors), aspect='auto')

n_colors = 2
plt.subplot(2, 3, 6)
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_colors), fontsize=10)
plt.imshow(Get_processed_image(image_array, n_colors), aspect='auto')
plt.savefig('my.png',dpi = 1000)
