FROM tensorflow/tensorflow:latest-gpu

COPY tf_gpu_test.py /src/tf_gpu_test.py

ENTRYPOINT [ "python", "/src/tf_gpu_test.py" ]
