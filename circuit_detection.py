import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency

# Load a pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dictionary to hold activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hook to capture activations from the last convolutional layer
# Register hooks for multiple layers
model.layer2[1].register_forward_hook(get_activation('layer2_block2'))
model.layer3[1].register_forward_hook(get_activation('layer3_block2'))
model.layer4[1].register_forward_hook(get_activation('layer4_block2'))
#model.layer4[1].register_forward_hook(get_activation('layer4_block2'))

def load_image(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

def visualize_feature_maps(activations, layer_name):
    layer_activations = activations[layer_name].squeeze(0).cpu().numpy()

    # Plot each feature map
    num_features = layer_activations.shape[0]
    plt.figure(figsize=(15, 15))
    for i in range(min(num_features, 16)):  # Display up to 16 feature maps
        plt.subplot(4, 4, i + 1)
        plt.imshow(layer_activations[i], cmap="viridis")
        plt.axis('off')
    plt.suptitle(f"Feature Maps in {layer_name}")
    
    # Save the feature maps figure
    plt.savefig(f"{layer_name}_feature_maps.png")
    plt.close()

def saliency_map(image_tensor, target_class):
    # Ensure requires_grad is set for saliency map computation
    image_tensor.requires_grad = True
    
    saliency = Saliency(model)
    saliency_map = saliency.attribute(image_tensor, target=target_class)
    saliency_map = saliency_map.squeeze().cpu().numpy()
    
    # Display and save the saliency map
    plt.imshow(saliency_map[0], cmap='hot')
    plt.title("Saliency Map")
    plt.axis('off')
    plt.savefig("saliency_map.png")
    plt.close()

def main():
    # Load and process the image
    image_path = 'images/bike.jpg'
    image_tensor = load_image(image_path)

    # Run the model and capture activations
    with torch.no_grad():
        output = model(image_tensor)

    # Example target class - modify based on your image
    target_class = output.argmax().item()
    print(f"Predicted Class: {target_class}")

    # Visualize and save feature maps for each layer in activations
    for layer_name in activations:
        visualize_feature_maps(activations, layer_name)

    # Generate a saliency map
    saliency_map(image_tensor, target_class)


if __name__ == "__main__":
    main()
