# tests/test_gpu.py

print("\n===== GPU TEST: TensorFlow =====")
try:
    import sys
    import os
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(str(PROJECT_ROOT)))
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU devices:", gpus)
    if gpus:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("Matmul result (TensorFlow, on GPU):\n", c.numpy())
        print("✅ TensorFlow bruger GPU! 🚀")
    else:
        print("❌ TensorFlow fandt INGEN GPU – bruger CPU.")
except Exception as e:
    print("Fejl i TensorFlow GPU-test:", e)

print("\n===== GPU TEST: PyTorch =====")
try:
    import torch
    print("PyTorch version:", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print("CUDA available:", cuda_avail)
    device_count = torch.cuda.device_count()
    print("Device count:", device_count)
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU name:", gpu_name)
        x = torch.rand(3, 3, device='cuda')
        y = torch.rand(3, 3, device='cuda')
        z = x + y
        print("GPU calculation success (PyTorch, on GPU):\n", z)
        print("✅ PyTorch bruger GPU! 🚀")
    else:
        print("❌ PyTorch fandt INGEN GPU – bruger CPU.")
except Exception as e:
    print("Fejl i PyTorch GPU-test:", e)

print("\n===== GPU test færdig =====\n")
