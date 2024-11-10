# Circuit Detection - Training 2

This repository provides a training exercise for detecting circuits in a neural network. You’ll load a pre-trained model, capture activations for specific layers, and apply clustering to identify circuits of neurons that activate together.

## Setup

1. **Clone the Repository**:

    git clone https://github.com/IdaCy/circuit-detection-training.git
    cd circuit-detection-training

2. **Install Required Packages**:

    pip install -r requirements.txt

3. **Download Sample Images!**

Add sample images to the images/ folder to test circuit detection on different inputs.

4. **Running Circuit Detection:**

Run the script to see the activations, clustering results, and visualizations for circuits:

    python cnn_circuits.py
    python lm_circuits.py

This will load the model, capture activations from specific layers, apply clustering, and visualize the identified circuits.


## Example images by courtesy of Pixabay:
https://pixabay.com/photos/bicycle-bike-activity-cycle-789648/
https://pixabay.com/vectors/bicycle-bike-cycling-transport-7876692/
https://pixabay.com/photos/cat-kitten-pet-striped-young-1192026/
https://pixabay.com/photos/woman-face-smile-lips-hairstyle-8592765/
https://pixabay.com/photos/glencoe-scotland-nature-landscape-8299076/
https://pixabay.com/photos/wood-boards-texture-wooden-brown-2045380/
