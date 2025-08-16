#!/usr/bin/env bash
set -euo pipefail

# Update
apt-get update

# Install OS-level deps required by Playwright browsers
# (names chosen to cover the libraries reported by Render)
apt-get install -y --no-install-recommends \
    libgtk-4-1 \
    libgraphene-1.0-0 \
    libgstreamer-gl1.0-0 \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-gl \
    libavif15 \
    libenchant-2-2 \
    libsecret-1-0 \
    libmanette-0.2-0 \
    libgles2-mesa \
    mesa-utils \
    ca-certificates \
    fonts-liberation \
    wget \
    unzip

# Install Python deps
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Playwright browsers (with additional system deps installer)
# using --with-deps helps, but the apt packages we installed earlier cover most libs
python -m playwright install --with-deps
