import tensorflow as tf

print(tf.__version__)
# 1.14.0

check_result = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

print(check_result)