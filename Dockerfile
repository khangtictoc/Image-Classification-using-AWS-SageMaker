# SageMaker PyTorch image
# List of available images
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2-cpu-py310

# Install package
RUN pip install -U smdebug==1.0.11 sagemaker torch torchvision

ENV PATH = "/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY scripts/train_model.py /opt/ml/code/train_model.py

# Defines cifar10.py as script entrypoint
ENV SAGEMAKER_PROGRAM train_model.py