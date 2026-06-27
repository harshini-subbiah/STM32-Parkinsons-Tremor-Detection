# convert_tflite.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

MODEL_IN = r"C:\tremor_project\model\keras_model.h5"
MODEL_OUT_FLOAT = r"C:\tremor_project\model\tremor_model_float.tflite"
MODEL_OUT_INT8 = r"C:\tremor_project\model\tremor_model_int8.tflite"
REP = r"C:\tremor_project\data\windows\features.npz"
NORM = r"C:\tremor_project\model\norm_stats.npz"

print("Loading keras model:", MODEL_IN)
model = keras.models.load_model(MODEL_IN, compile=False)
print("Model loaded. TF:", tf.__version__)

# Save float32 tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
with open(MODEL_OUT_FLOAT, "wb") as f:
    f.write(tflite_float)
print("Saved float32 TFLite:", MODEL_OUT_FLOAT)

# Prepare representative dataset for int8 (if you want full int8)
rep = np.load(REP)
Xrep = rep['X']  # (n, features)
ns = np.load(NORM)
mean = ns['mean']; std = ns['std']

def rep_gen():
    for i in range(min(300, Xrep.shape[0])):
        x = ((Xrep[i:i+1] - mean) / (std + 1e-9)).astype(np.float32)
        yield [x]

# Full integer quantization (input/output int8)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    with open(MODEL_OUT_INT8, "wb") as f:
        f.write(tflite_int8)
    print("Saved INT8 TFLite:", MODEL_OUT_INT8)
except Exception as e:
    print("INT8 conversion failed:", e)
    print("You can still use float32 TFLite or try dynamic/float16 quantization.")
