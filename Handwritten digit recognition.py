from os import access
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.activations import softmax
data=tf.keras.datasets.mnist
from tensorflow.python.keras.metrics import accuracy

(x_train,y_train),(x_test,y_test)=data.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

# Create a figure to display all images in a grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Handwritten Digit Recognition Results', fontsize=16, fontweight='bold')
axes = axes.flatten()

for x in range(1, 5):
    # Read image with correct filename format
    img_path = f'{x} png.png'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        axes[x-1].axis('off')
        continue
    
    # Resize to 28x28 if needed (MNIST format)
    if img.shape != (28, 28):
        img = cv.resize(img, (28, 28))
    
    # Invert if needed (MNIST has white digits on black background)
    # If your images are black digits on white, uncomment the next line:
    # img = 255 - img
    
    # Normalize to 0-1 range
    img_normalized = img.astype(np.float32) / 255.0
    
    # Reshape for model prediction
    img_array = np.array([img_normalized])
    prediction = model.predict(img_array, verbose=0)
    
    print(" ---------- ")
    print(f"Image: {img_path}")
    print("The predicted output is:", np.argmax(prediction))
    print(f"Confidence: {np.max(prediction)*100:.2f}%")
    print(" ---------- ")
    
    # Display the image with white background (inverted colormap)
    # Invert the image for display: white background, black digits
    img_display = 1.0 - img_normalized
    axes[x-1].imshow(img_display, cmap='gray', vmin=0, vmax=1)
    axes[x-1].set_title(f'Image {x}: Predicted = {np.argmax(prediction)}\nConfidence: {np.max(prediction)*100:.2f}%', 
                        fontsize=12, fontweight='bold')
    axes[x-1].axis('off')

plt.tight_layout()
plt.show()
    
