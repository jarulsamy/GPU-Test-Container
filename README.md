# GPU Test Container

A simple Python script to test GPU capabilities from a container. The current
version checks if Tensorflow can (1) utilize a GPU at all, and (2) performs a
simple matrix multiplication to ensure everything is working correctly.

## Running

### Docker

A `Dockerfile` is provided. Build the image and run it.

```cli
$ docker build . -t gpu_test:latest
. . .
$ docker run --gpus all --rm gpu_test:latest
```

> Note: **nvidia-docker** v2 uses **--runtime=nvidia** instead of **--gpus
> all**. **nvidia-docker** v1 uses the **nvidia-docker** alias, rather than the
> **--runtime=nvidia** or **--gpus all** command line flags.

You should see something similar to this

```text
2023-06-23 17:14:22.598212: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-06-23 17:14:24.387920: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3830 MB memory: -> device: 0, name: Quadro P2000, pci bus id: 0000:65:00.0, compute capability: 6.1
2023-06-23 17:14:24.398485: I tensorflow/core/common_runtime/placer.cc:114] input: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.398517: I tensorflow/core/common_runtime/placer.cc:114] \_EagerConst: (\_EagerConst): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.398529: I tensorflow/core/common_runtime/placer.cc:114] output_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.421968: I tensorflow/core/common_runtime/eager/execute.cc:1525] Executing op \_EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.422565: I tensorflow/core/common_runtime/eager/execute.cc:1525] Executing op \_EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.423578: I tensorflow/core/common_runtime/placer.cc:114] a: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.423600: I tensorflow/core/common_runtime/placer.cc:114] b: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.423615: I tensorflow/core/common_runtime/placer.cc:114] MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.423629: I tensorflow/core/common_runtime/placer.cc:114] product_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.424436: I tensorflow/core/common_runtime/eager/execute.cc:1525] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2023-06-23 17:14:24.631235: I tensorflow/core/common_runtime/placer.cc:114] a: (\_Arg): /job:localhost/replica:0/task:0/device:CPU:0
2023-06-23 17:14:24.631270: I tensorflow/core/common_runtime/placer.cc:114] b: (\_Arg): /job:localhost/replica:0/task:0/device:CPU:0
2023-06-23 17:14:24.631286: I tensorflow/core/common_runtime/placer.cc:114] MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
2023-06-23 17:14:24.631299: I tensorflow/core/common_runtime/placer.cc:114] product_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:CPU:0
2023-06-23 17:14:24.631900: I tensorflow/core/common_runtime/eager/execute.cc:1525] Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
================================================================================
Num GPUs Available: 1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
================================================================================
Let us do a simple matrix multiplication on the GPU to verify...
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
================================================================================
Let us repeat that test on the CPU...
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
================================================================================
input: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
\_EagerConst: (\_EagerConst): /job:localhost/replica:0/task:0/device:GPU:0
output_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:GPU:0
a: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
b: (\_Arg): /job:localhost/replica:0/task:0/device:GPU:0
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
product_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:GPU:0
a: (\_Arg): /job:localhost/replica:0/task:0/device:CPU:0
b: (\_Arg): /job:localhost/replica:0/task:0/device:CPU:0
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0
product_RetVal: (\_Retval): /job:localhost/replica:0/task:0/device:CPU:0
```

### Apptainer (formerly Singularity)

Convert the Docker image to an Apptainer `.sif` then run the resulting
container.

1. Build the Docker image:

   ```cli
   $ docker build . -t gpu_test:latest
   ```

2. Convert it to a Apptainer `.cif`

   ```cli
   $ apptainer build gpu_test.sif docker-daemon:gpu_test:latest
   INFO:    Starting Build...
   . . .
   INFO:    Creating SIF file...
   INFO:    Build complete: gpu_test.sif
   ```

   You should have a `gpu_test.sif` file in your current working directory.

   ```cli
   $ ls gpu_test.sif
   gpu_test.sif
   ```

3. Run the resulting image.

   ```cli
   $ apptainer run
   ```

   The output should be the same as the Docker run output.
