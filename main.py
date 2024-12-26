"""
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

image = load_image(r"D:\Downloads\test -4 (boys all).jpg")
mtcnn = MTCNN()
result = mtcnn.detect_faces(image)
plt.imshow(plot(image,result))
plt.show()
"""

import tensorflow_hub as hub

model = hub.load('https://tfhub.dev/google/facenet/1')
print("Model Loaded Successfully")
