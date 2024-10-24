# Use a more recent PyTorch base image with Python >= 3.8
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# Create a group and user for better security practices
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

# Create necessary directories and set ownership
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

# Switch to the non-root user
USER algorithm

# Set the working directory
WORKDIR /opt/algorithm

# Ensure that the user's local bin is in the PATH
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Upgrade pip for the user
RUN python -m pip install --user -U pip

# Copy all required files into the Docker image
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm model.pth /opt/algorithm/
COPY --chown=algorithm:algorithm model.pth /home/algorithm/.cache/torch/hub/checkpoints/model.pth
COPY --chown=algorithm:algorithm training_utils /opt/algorithm/training_utils

# Install required Python packages via pip
RUN python -m pip install --user -r requirements.txt

# Copy additional Python scripts
COPY --chown=algorithm:algorithm process.py postprocessing.py /opt/algorithm/

# Set the entrypoint to execute the script
ENTRYPOINT ["bash", "entrypoint.sh"]

# Define necessary labels for the algorithm
LABEL nl.diagnijmegen.rse.algorithm.name=noduledetection
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=12G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G
