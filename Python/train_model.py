# train_model.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

FEAT = r'C:\tremor_project\data\windows\features.npz'
MODEL_OUT = r'C:\tremor_project\model\keras_model.h5'

data = np.load(FEAT)
X = data['X']   # shape (n, 13)
y = data['y']

# normalize
ns = np.load(r'C:\tremor_project\model\norm_stats.npz')
mean = ns['mean']; std = ns['std']
Xn = (X - mean) / (std + 1e-9)

# shuffle and split
idx = np.arange(len(Xn)); np.random.seed(42); np.random.shuffle(idx)
Xn = Xn[idx]; y = y[idx]
split = int(0.8*len(Xn))
Xtr, Xte = Xn[:split], Xn[split:]
ytr, yte = y[:split], y[split:]

num_classes = len(np.unique(y))
ytr_cat = keras.utils.to_categorical(ytr, num_classes)
yte_cat = keras.utils.to_categorical(yte, num_classes)

# model (tiny)
model = keras.Sequential([
    layers.Input(shape=(Xtr.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(Xtr, ytr_cat, validation_data=(Xte,yte_cat), epochs=200, batch_size=32, callbacks=[es])

# save
model.save(MODEL_OUT, include_optimizer=False)
print("Saved model to:", MODEL_OUT)
