# Import libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=64)

# Evaluate
model.evaluate(X_test, y_test)

# Predictions
sample_images = X_test[:5]
sample_labels = y_test[:5]
predictions = model.predict(sample_images)
result = np.argmax(predictions, axis=1)

# Display results
for i, img in enumerate(sample_images):
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {np.argmax(sample_labels[i])}\nPred: {result[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
