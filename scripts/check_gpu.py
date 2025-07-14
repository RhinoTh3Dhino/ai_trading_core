import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("Build:", tf.test.is_built_with_cuda())
