# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Add labels for metadata
LABEL Maintainer="P10-Energy" \
      Project="P10 Energy Container"

# Install additional dependencies
RUN echo "Installing P9 Energy dependencies" && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install pandas numpy scikit-learn matplotlib seaborn holidays optuna neuralforecast lightning && \
    echo "P10 Energy dependencies installed"

# Test the setup
RUN echo "Testing P10 Energy Container" && \
    python3 --version && \
    pip3 --version && \
    echo "P10 Energy Container is ready to use"

# Set the default command to execute when the container starts
CMD ["bash"]
