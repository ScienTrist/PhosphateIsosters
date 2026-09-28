#!/bin/bash

# Update package lists
sudo apt update

# Install Python and pip
sudo apt install -y python3 python3-pip wget g++ build-essential

# Install dependencies from requirements.txt
pip3 install --break-system-packages -r requirements.txt

# Download and compile TMalign (optional, but convenient)
if [ ! -f "TMalign" ]; then
    echo "Downloading TMalign source..."
    wget https://zhanggroup.org/TM-align/TMalign.cpp
    echo "Compiling TMalign..."
    g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
    rm TMalign.cpp
    echo "TMalign installed successfully!"
fi

echo "Setup complete. You can now run the script using: python3 main.py"
