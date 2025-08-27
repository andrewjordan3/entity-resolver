#!/bin/bash

# ==============================================================================
# Setup Script for entity-resolver in Google Colab
# ==============================================================================
#
# This script automates the environment setup for the entity-resolver project
# within a Google Colab notebook. It performs two main tasks:
#
# 1. Installs the RAPIDS suite (cuDF, cuML, etc.) using the official
#    utility script. This is a special installation process required for
#    Colab and cannot be done via a standard requirements.txt file.
#
# 2. Installs all standard, CPU-based Python dependencies listed in the
#    `requirements.txt` file.
#
# To run this script in Colab:
# 1. Make it executable: !chmod +x setup_colab.sh
# 2. Execute it:       !./setup_colab.sh
#
# ==============================================================================

# --- Script Configuration ---
# The `set -e` command ensures that the script will exit immediately if any
# command fails. This prevents a partial or broken installation.
set -e

# --- Step 1: Install RAPIDS Libraries ---
echo "========================================="
echo "STEP 1: Installing RAPIDS Libraries..."
echo "========================================="

# The RAPIDS team provides a utility repository to handle the complex
# installation process within Colab. We clone this repository first.
echo "--> Cloning RAPIDS installer repository..."
git clone https://github.com/rapidsai/rapidsai-csp-utils.git

# Navigate into the cloned repository to access the installer script.
cd rapidsai-csp-utils

# Execute the Python script provided by NVIDIA to install the correct
# versions of cuDF, cuML, and other RAPIDS libraries that are compatible
# with the Colab environment.
echo "--> Running the RAPIDS installation script (this may take several minutes)..."
python ./colab/install_rapids.py

# Navigate back to the root directory of your project.
cd ..
echo "--> RAPIDS installation complete."
echo ""


# --- Step 2: Install Core CPU Libraries ---
echo "========================================="
echo "STEP 2: Installing Core Python Libraries..."
echo "========================================="

# Now that the special GPU libraries are installed, we can install all the
# standard, platform-agnostic Python packages using pip and the
# requirements.txt file.
echo "--> Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "--> Core library installation complete."
echo ""


# --- Finalization ---
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "IMPORTANT: Please restart the Colab runtime now for all changes to take effect."
echo "Go to 'Runtime' -> 'Restart runtime' in the Colab menu."
