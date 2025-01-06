# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11.10
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /media/research/DataStrorage/johnk/monst3r

# Install required system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# Explicitly set HOME environment variable
ENV HOME=/home/appuser

# Create .evo directory in the user's home
RUN mkdir -p /home/appuser/.evo && chown -R appuser:appuser /home/appuser

# Copy the requirements and local dependencies first
COPY requirements.txt .
COPY third_party/ ./third_party/

# # Download dependencies as a separate step to take advantage of Docker's caching.
# # Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# # Leverage a bind mount to requirements.txt to avoid having to copy them into
# # into this layer.
# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=requirements.txt,target=requirements.txt \
#     python -m pip install -r requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model checkpoint - add this section
COPY checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth ./checkpoints/



# Copy the source code into the container.
COPY . .

# Give ownership of the project directory to appuser
RUN chown -R appuser:appuser /media/research/DataStrorage/johnk/monst3r

# Switch to the non-privileged user to run the application.
USER appuser


# Expose the port that the application listens on.
EXPOSE 7860 8000

# Run the application.
CMD python demo.py --device cuda
