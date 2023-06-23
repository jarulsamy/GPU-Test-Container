#!/usr/bin/env python3
"""A simple test which loads tensorflow and checks if the GPU is accessible."""

import argparse
from pprint import pprint

import tensorflow as tf

parser = argparse.ArgumentParser(
    description=__doc__,
)
parser.add_argument(
    "--version",
    action="version",
    version="%(prog)s 1.0",
)
parser.parse_args()

LINE = "=" * 80

tf.debugging.set_log_device_placement(True)

gpu_devices = tf.config.list_physical_devices("GPU")

print(LINE)

print(f"Num GPUs Available: {len(gpu_devices)}")
pprint(gpu_devices)

print(LINE)

print("Let us do a simple matrix multiplication on the GPU to verify...")
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

print(LINE)
print("Let us repeat that test on the CPU...")
with tf.device("/CPU:0"):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)

print(LINE)
