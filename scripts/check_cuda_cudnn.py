import ctypes
import os

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Build:", tf.test.is_built_with_cuda())

print("PATH:", os.environ["PATH"])
try:
    ctypes.WinDLL("cudnn64_8.dll")
    print("cuDNN DLL found!")
except Exception as e:
    print("cuDNN DLL NOT found!", e)
