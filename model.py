import numpy as np
import keras
from sklearn.model_selection import train_test_split
from functions import load_digits
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import BatchNormalization, Dropout
from keras.models import Sequential
import pickle
X = []
y = []
data = load_digits()

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 0-255 to 0-1
X_test = X_test / 255
X_train = X_train / 255

train_y_one_hot = keras.utils.to_categorical(y_train)
test_y_one_hot = keras.utils.to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], 40, 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 40, 1)

model = Sequential()
model.add(
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(40, 40, 1)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# fit model
model.summary()
history = model.fit(X_train, train_y_one_hot, epochs=10, batch_size=32, validation_data=(X_test, test_y_one_hot),
                    verbose=1)

model.save('model.h5')
