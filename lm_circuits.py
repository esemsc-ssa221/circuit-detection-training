import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to capture activations and attentions for a given sentence
def get_model_outputs(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    attentions = outputs.attentions  # List of attention tensors for each layer
    hidden_states = outputs.hidden_states  # List of hidden state tensors for each layer
    return attentions, hidden_states

# Apply PCA and K-means to find circuits
def find_circuits(hidden_states, layer_index, n_components=10, n_clusters=5):
    """
    Apply PCA and K-means clustering on hidden states for a given layer.
    
    Parameters:
        hidden_states (list): List of hidden state tensors for each layer.
        layer_index (int): Index of the layer to analyze.
        n_components (int): Desired number of PCA components.
        n_clusters (int): Number of clusters to find with K-means.
        
    Returns:
        cluster_labels (np.array): Array with cluster assignments for each neuron.
    """
    layer_activations = hidden_states[layer_index].squeeze(0).detach().cpu().numpy()  # Shape: (seq_len, hidden_dim)

    # Flatten across the sequence length
    flattened_activations = layer_activations.reshape(-1, layer_activations.shape[-1])

    # Set n_components dynamically based on data size
    n_components = min(n_components, flattened_activations.shape[0], flattened_activations.shape[1])

    # Dimensionality reduction using PCA
    pca = PCA(n_components=n_components)
    reduced_activations = pca.fit_transform(flattened_activations)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(reduced_activations)

    # Print cluster information
    print(f"Layer {layer_index}: Circuit Clusters Distribution - {np.bincount(cluster_labels)}")
    return cluster_labels, reduced_activations

# Main function to run circuit detection
def main():
    # Test multiple sentences to compare circuit activation
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells sea shells by the sea shore.",
        "Artificial intelligence is transforming the world."
    ]

    for sentence in sentences:
        print(f"Analyzing sentence: '{sentence}'")
        # Get model outputs for the current sentence
        attentions, hidden_states = get_model_outputs(sentence)

        # Analyze circuits for a specific layer
        layer_index = 1
        cluster_labels, reduced_activations = find_circuits(hidden_states, layer_index)
        print(f"Cluster labels for layer {layer_index}: {cluster_labels}\n")

        # Call this function inside your loop for each sentence
        visualize_clusters(reduced_activations, cluster_labels, sentence, f"Layer {layer_index}")


def visualize_clusters(reduced_activations, cluster_labels, sentence, layer_name):
    # Dynamically set perplexity to be less than the number of samples
    n_samples = reduced_activations.shape[0]
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    tsne_result = tsne.fit_transform(reduced_activations)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Cluster Visualization for {layer_name}\nSentence: {sentence}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Create a filename based on the layer and a shortened version of the sentence
    short_sentence = "_".join(sentence.split()[:5])  # Use the first 5 words as part of the filename
    filename = f"{layer_name}_{short_sentence}.png"
    
    # Save the plot
    output_dir = "cluster_visualizations"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    main()
