# %% [raw]
# ---
# aliases: ["/what-are-image-embeddings/"]
# categories: ["Computer Vision", "Machine Learning"]
# date: "2025-03-16"
# image: "/images/what_are_image_embeddings/thumbnail.png"
# title: "What are Image Embeddings?"
# subtitle: "Understanding how images are represented as numerical vectors for AI applications"
# format: "html"
# ---

# %% [markdown]
# This notebook explores the concept of image embeddings, how they work, and their applications in AI. We'll focus on Google's SigLIP 2, a state-of-the-art multilingual vision-language encoder, and demonstrate its practical applications through visualization, clustering, and text-image similarity analysis.

# %% [markdown]
# ## Introduction
# Image embeddings are numerical representations of images that capture their semantic content in a way that's useful for machine learning algorithms. At their core, embeddings are dense vectors—typically consisting of hundreds or thousands of floating-point numbers—that represent images in a high-dimensional space where similar images are positioned close to each other.
# 
# ### Why Do We Need Image Embeddings?
# 
# Images in their raw pixel form are:
# 
# - **High-dimensional**: A 224x224 RGB image contains 150,528 pixel values
# - **Not semantically organized**: Similar-looking images might have very different pixel values
# - **Difficult to work with**: Comparing raw pixels doesn't capture semantic similarity
# 
# Embeddings solve these problems by:
# 
# - **Reducing dimensionality**: Typically to a few hundred or thousand dimensions
# - **Capturing semantics**: Images with similar content have similar embeddings
# - **Enabling efficient search**: Finding similar images becomes a vector similarity search
# - **Supporting transfer learning**: Pre-trained embeddings can be used for various downstream tasks

# %% [markdown]
# ## How Image Embeddings Work
# 
# Modern image embeddings are typically created using deep neural networks, particularly convolutional neural networks (CNNs) or vision transformers (ViTs). These networks learn to transform raw pixels into compact, semantically meaningful representations through extensive training on large datasets.
# 
# The process generally involves:
# 
# 1. **Training**: Neural networks are trained on large image datasets, often using self-supervised or weakly-supervised learning approaches
# 2. **Feature extraction**: The trained network processes an image through its layers
# 3. **Embedding generation**: The network's final or penultimate layer outputs become the embedding vector
# 
# These embeddings can then be used for various tasks:
# 
# - **Image similarity**: Finding visually or semantically similar images
# - **Image classification**: Categorizing images into predefined classes
# - **Image retrieval**: Finding relevant images based on text queries
# - **Zero-shot learning**: Recognizing objects the model wasn't explicitly trained on
# - **Transfer learning**: Using pre-trained embeddings for new tasks with limited data

# %% [markdown]
# ## SigLIP 2: Google's Advanced Multilingual Vision-Language Encoder
# 
# SigLIP 2 represents the latest advancement in image embedding technology. Developed by Google and released in early 2024, it significantly improves upon its predecessor by offering enhanced semantic understanding, better localization capabilities, and more effective dense feature representation.

# %% [markdown]
# ### Technical Background and Evolution
# 
# #### From CLIP to SigLIP to SigLIP 2
# 
# Vision-language models have evolved considerably in recent years:
# 
# 1. **CLIP and ALIGN**: These pioneered the approach of jointly training image and text encoders to understand the semantic relationship between visual data and natural language
# 
# ![Contrast function comparison between CLIP and SigLIP](https://i.imgur.com/GH9sai5.png)
# 
# 2. **SigLIP (1st generation)**: Improved upon CLIP by replacing its contrastive loss function with a simpler pairwise sigmoid loss. Instead of requiring a global view of pairwise similarities for normalization (as in contrastive learning), the sigmoid loss operated only on image-text pairs, allowing for better scaling and improved performance even with smaller batch sizes
# 
# 3. **SigLIP 2**: Extends this foundation by incorporating several additional training techniques into a unified recipe, creating more powerful and versatile vision-language encoders that outperform their predecessors across all model scales

# %% [markdown]
# ### How SigLIP 2 Works
# 
# #### Enhanced Training Methodology
# 
# SigLIP 2's functioning is fundamentally based on its innovative training approach that combines multiple previously independent techniques:
# 
# 1. **Extended Training Objectives**: While preserving the original sigmoid loss function, SigLIP 2 integrates several additional training objectives:
#    - Captioning-based pretraining to enhance semantic understanding
#    - Self-supervised losses including self-distillation and masked prediction
#    - Online data curation for improved quality and diversity of training examples
# 
# 2. **Multilingual Capabilities**: The model is trained on a more diverse data mixture that incorporates de-biasing techniques, leading to significantly better multilingual understanding and improved fairness across different languages and cultures
# 
# 3. **Technical Implementation**: SigLIP 2 models use the Gemma tokenizer with a vocabulary size of 256,000 tokens, allowing for better representation of diverse languages

# %% [markdown]
# #### Beyond Simple Cosine Similarity: Advanced Similarity Computation
# 
# While many discussions of image embeddings focus on simple cosine similarity between vectors, SigLIP 2's similarity computation is actually much more sophisticated. This advanced approach leads to more accurate and nuanced similarity scores:
# 
# 1. **Multi-head Attention Pooling (MAP)**: Unlike simpler models that use average pooling to aggregate token representations, SigLIP 2 employs a more sophisticated attention-based pooling mechanism:
#    - The MAP head learns to focus on the most relevant parts of the image or text
#    - It assigns different weights to different regions or tokens based on their importance
#    - This selective attention mechanism produces more contextually relevant embeddings that capture important details while ignoring noise
# 
# 2. **Temperature Scaling**: SigLIP 2 applies a learned temperature parameter (τ) to scale similarity scores:
#    - Raw cosine similarities are divided by this temperature: sim(i,j)/τ
#    - Lower temperature values make the distribution more "peaked," emphasizing differences between high and low similarity pairs
#    - Higher temperature values make the distribution more uniform
#    - The temperature parameter is learned during training to optimize the model's discrimination ability
# 
# 3. **Bias Term Adjustment**: The similarity calculation includes a learned bias term:
#    - sim'(i,j) = sim(i,j)/τ + b, where b is the learned bias
#    - This bias helps counteract the inherent imbalance between positive and negative pairs during training
#    - It acts as a calibration factor, adjusting the similarity scores to better reflect true semantic relationships
# 
# 4. **Sigmoid Activation**: Unlike models that use softmax normalization (like CLIP), SigLIP 2 applies a sigmoid function to the adjusted similarity scores:
#    - p(i,j) = sigmoid(sim'(i,j)) = 1/(1+exp(-(sim(i,j)/τ + b)))
#    - This transforms the unbounded similarity scores into well-calibrated probability-like values in the range [0,1]
#    - The sigmoid function allows each image-text pair to be evaluated independently, which is more appropriate for retrieval tasks
# 
# These components work together to ensure that SigLIP 2's similarity calculations go far beyond simple vector dot products. When using SigLIP 2, it's crucial to use the model's built-in comparison mechanism (`logits_per_image` followed by sigmoid activation) rather than manually computing cosine similarity on raw embeddings, as the former incorporates all these learned parameters and transformations that were optimized during training.

# %% [markdown]
# #### Architecture Variants
# 
# SigLIP 2 is available in several architectural variants to accommodate different computational constraints and use cases:
# 
# 1. **Model Sizes**: The family includes four primary model sizes:
#    - ViT-B (86M parameters)
#    - ViT-L (303M parameters)
#    - ViT-So400m (400M parameters)
#    - ViT-g (1B parameters)
# 
# 2. **NaFlex Variants**: One of the most significant innovations in SigLIP 2 is the introduction of NaFlex variants, which support dynamic resolution and preserve the input's native aspect ratio. This feature is particularly valuable for:
#    - Optical character recognition (OCR)
#    - Document understanding
#    - Any task where preserving the original aspect ratio and resolution is important

# %% [markdown]
# ### Key Capabilities and Improvements
# 
# SigLIP 2 models demonstrate significant improvements over the original SigLIP across several dimensions:
# 
# 1. **Core Capabilities**: The models outperform their SigLIP counterparts at all scales in:
#    - Zero-shot classification
#    - Image-text retrieval
#    - Transfer performance when used for visual representation in Vision-Language Models (VLMs)
# 
# 2. **Localization and Dense Features**: The enhanced training recipe leads to substantial improvements in localization and dense prediction tasks, making the models more effective for detailed visual understanding
# 
# 3. **Multilingual Understanding**: Through its diverse training data and de-biasing techniques, SigLIP 2 achieves much better multilingual understanding and improved fairness compared to previous models

# %% [markdown]
# ### Practical Applications
# 
# The improvements in SigLIP 2 make it particularly well-suited for:
# 
# 1. **Zero-shot Image Classification**: Using the model to classify images into categories it wasn't explicitly trained on
# 
# 2. **Image-Text Retrieval**: Finding relevant images based on text queries or finding appropriate textual descriptions for images
# 
# 3. **Feature Extraction for VLMs**: Providing high-quality visual representations that can be combined with large language models to build more capable vision-language models
# 
# 4. **Document and Text-Heavy Image Analysis**: Particularly with the NaFlex variants, which excel at tasks requiring preservation of aspect ratio and resolution

# %% [markdown]
# ## Practical Applications of Image Embeddings
# 
# Now that we understand the theoretical background of image embeddings, let's explore their practical applications. Image embeddings form the foundation for numerous computer vision tasks and enable powerful capabilities like semantic search, clustering, and cross-modal understanding.
# 
# ### Key Applications of Image Embeddings
# 
# 1. **Visual Similarity Search**: Find visually similar images based on embedding distance
# 2. **Image Clustering**: Group images by semantic content without explicit labels
# 3. **Cross-Modal Understanding**: Connect images with text descriptions
# 4. **Fine-Grained Recognition**: Identify specific attributes and details
# 5. **Transfer Learning**: Apply pre-trained embeddings to new, domain-specific tasks
# 
# SigLIP 2, with its powerful multilingual capabilities and improved semantic understanding, enables these applications with state-of-the-art performance. While SigLIP 2 comes in various sizes (Base, Large, So400m, and Giant) and configurations, we'll focus on the So400m model, which provides an excellent balance of quality and efficiency.

# %% [markdown]
# ## Implementing SigLIP 2: Practical Examples
# 
# Now that we understand the theoretical background of image embeddings and SigLIP 2, let's implement it to see how it works in practice. We'll use the Hugging Face Transformers library, which provides easy access to SigLIP 2 models.

# %%
# Import necessary libraries
import sys
import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModel, AutoProcessor
from transformers.image_utils import load_image

# %% [markdown]
# ### Loading the SigLIP 2 Model
# 
# We'll use the So400m variant of SigLIP 2 for our examples, which offers an excellent balance of quality and efficiency. The most recent models are available with the "google/siglip2-" prefix.

# %%
# We'll use the SO400M model which offers good performance
model_name = "google/siglip2-so400m-patch14-384"

# Define a function to extract embeddings from an image
def get_image_embedding(image_path_or_url, model, processor):
    """Extract embeddings from an image file or URL
    
    NOTE: For most SigLIP applications, you should NOT extract embeddings separately.
    Instead, use the model to process image-text pairs together via model(**inputs)
    to get direct similarity scores through the model's logits_per_image.
    
    This function is provided for educational purposes or for specific use cases
    where you need the raw embeddings.
    """
    # Load image from URL or local path
    if isinstance(image_path_or_url, str):
        if image_path_or_url.startswith(('http://', 'https://')):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
    else:
        # Assuming it's already a PIL Image
        image = image_path_or_url
    
    # Process image and extract embedding
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Just get image features directly
        image_embedding = model.get_image_features(**inputs)
        image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    
    return image_embedding.squeeze().detach().numpy(), image

# %% [markdown]
# ### Example 1: Zero-Shot Image Classification
# 
# Let's use SigLIP 2 for zero-shot image classification. We'll load an image and classify it against different text prompts.

# %%
# Set up the zero-shot classification pipeline
from transformers import pipeline
from PIL import Image
import requests
import matplotlib.pyplot as plt

# SigLIP 2 uses the Gemma tokenizer which requires specific parameters
pipe = pipeline(
    model=model_name, 
    task="zero-shot-image-classification",
)

inputs = {
    "images": [
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg", # bear
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg", # teddy bear
    ],
    "texts": [
        "bear looking into the camera",
        "bear looking away from the camera",
        "a bunch of teddy bears",
        "two teddy bears",
        "three teddy bears"
    ],
}

# Load images for display
display_images = []
for img_url in inputs["images"]:
    img = Image.open(requests.get(img_url, stream=True).raw)
    display_images.append(img)

outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])

# Display the outputs
for i, output in enumerate(outputs):
    print(f"Image {i+1} results:")
    for result in output:
        print(f"{result['label']}: {result['score']:.4f}")
    print()

# Visualize the results with images on top
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 2]})

# Display the images in the top row
for i, img in enumerate(display_images):
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Image {i+1}")
    axes[0, i].axis('off')

# Display the classification results in the bottom row
for i, output in enumerate(outputs):
    labels = [result['label'] for result in output]
    scores = [result['score'] for result in output]
    
    axes[1, i].bar(range(len(labels)), scores)
    axes[1, i].set_xticks(range(len(labels)))
    axes[1, i].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_title(f"Image {i+1} Classification Results")
    axes[1, i].set_ylabel("Probability")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Example 2: Image-Text Similarity
# 
# Now let's explore how we can use SigLIP 2 to compute similarity between multiple images and texts.

# %%
# Load the model and processor
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Define a set of sample images from COCO dataset for demonstration
image_urls = [
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg",  # bear
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000632.jpg",  # train
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000724.jpg",  # umbrella
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg",  # teddy bear
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000785.jpg",  # clock
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000802.jpg",  # train
]

# Extract embeddings and store images
embeddings = []
images = []
for i, url in enumerate(image_urls[:3]):  # Limiting to first 3 images to save time
    print(f"Processing image {i+1}/{len(image_urls[:3])}: {url}")
    embedding, image = get_image_embedding(url, model, processor)
    embeddings.append(embedding)
    images.append(image)

# Convert to numpy array for further processing
embeddings = np.array(embeddings)
print(f"Embedded {len(embeddings)} images. Embedding shape: {embeddings.shape}")

# Display the images
fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
for i, (image, ax) in enumerate(zip(images, axes)):
    ax.imshow(image)
    ax.set_title(f"Image {i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Text descriptions
texts = [
    "a wild bear",
    "a train on tracks",
    "a person with an umbrella",
    "a child's toy",
    "a stop sign",
    "a picture of a bedroom",
    "Cozy bedroom retreat filled with books, plants, and warm natural light",
    "a picture of a timepiece",
    "a picture of a vehicle for transportation"
]

# Get text embeddings using the processor and model
def get_text_embedding(text, model, processor):
    """Extract text embedding from a text string
    
    NOTE: For most SigLIP applications, you should NOT extract embeddings separately.
    Instead, use the model to process image-text pairs together via model(**inputs)
    to get direct similarity scores through the model's logits_per_image.
    
    This function is provided for educational purposes or for specific use cases
    where you need the raw embeddings.
    """
    inputs = processor(text=text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Just get text features directly
        text_embedding = model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)
    
    return text_embedding.squeeze().detach().numpy()

# Get embeddings for the text queries
text_embeddings = []
for i, query in enumerate(texts):
    print(f"Processing text {i+1}/{len(texts)}: '{query}'")
    text_embeddings.append(get_text_embedding(query, model, processor))
text_embeddings = np.array(text_embeddings)
print(f"Embedded {len(text_embeddings)} text queries. Embedding shape: {text_embeddings.shape}")
print("NOTE: While we extracted text embeddings separately, for similarity calculations")
print("we'll use the model's native capability to process image-text pairs together")

# %%
# Compute similarity between our images and texts
# Instead of computing dot product manually, let's use the model's built-in functionality

# Create a function to compute similarity between images and texts using the model directly
def compute_image_text_similarity(images, texts, model, processor):
    """Compute similarity between images and texts using the model's native capabilities"""
    similarity_matrix = np.zeros((len(images), len(texts)))
    
    for i, image in enumerate(images):
        # Process each image with all text descriptions
        inputs = processor(
            text=texts, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=64
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # The model directly computes logits_per_image which represents similarity
            logits = outputs.logits_per_image
            # Convert to probabilities
            probs = torch.sigmoid(logits)
            
            # Store the similarity scores for this image
            similarity_matrix[i] = probs[0].detach().numpy()
    
    return similarity_matrix

# Compute similarity using the model's native capabilities
print("Computing image-text similarity using the model's built-in functionality...")
similarity_matrix = compute_image_text_similarity(images, texts, model, processor)
print("Similarity computation complete.")

# Display similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, vmin=0, vmax=1, cmap='viridis')
plt.colorbar(label='Similarity Score')
plt.xticks(np.arange(len(texts)), texts, rotation=45, ha='right')
plt.yticks(np.arange(len(images)), [f"Image {i+1}" for i in range(len(images))])
plt.title('Image-Text Similarity Matrix')

# Add text annotations with the score values
for i in range(len(images)):
    for j in range(len(texts)):
        plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                 ha='center', va='center', 
                 color='white' if similarity_matrix[i, j] < 0.5 else 'black')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Example 3: Visualizing Embeddings with Clustering
# 
# Let's use clustering to group our images based on their semantic content. For a more meaningful analysis, we'll use a larger set of images from the COCO dataset and visualize them using UMAP before clustering.

# %%
# Import additional libraries for enhanced visualization
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from tqdm.notebook import tqdm

# Define a larger set of sample images from COCO dataset
coco_image_urls = [
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg",  # bear
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000632.jpg",  # train
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000724.jpg",  # umbrella
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg",  # teddy bear
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000785.jpg",  # clock
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000802.jpg",  # train
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000872.jpg",  # person with umbrella
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000885.jpg",  # dining table
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000934.jpg",  # person
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001000.jpg",  # zebra
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001296.jpg",  # sheep
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001425.jpg",  # airplane
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001490.jpg",  # giraffe
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001503.jpg",  # bird
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001532.jpg",  # dog
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001584.jpg",  # boat
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001675.jpg",  # person on bike
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001761.jpg",  # cat
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000001818.jpg",  # horse
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000002153.jpg",  # car
]

# Extract embeddings for all images
print("Extracting embeddings for all images...")
large_embeddings = []
large_images = []

for i, url in enumerate(tqdm(coco_image_urls)):
    try:
        embedding, image = get_image_embedding(url, model, processor)
        large_embeddings.append(embedding)
        large_images.append(image)
    except Exception as e:
        print(f"Error processing image {i+1}: {e}")

# Convert to numpy array
large_embeddings = np.array(large_embeddings)
print(f"Successfully embedded {len(large_embeddings)} images. Embedding shape: {large_embeddings.shape}")

# %% [markdown]
# ### Visualizing High-Dimensional Embeddings with UMAP

# %%
# Apply UMAP for dimensionality reduction to visualize embeddings in 2D
print("Applying UMAP dimensionality reduction...")
umap_model = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(large_embeddings)

# Function to plot images on UMAP projection
def plot_images_on_umap(embeddings_2d, images, figsize=(20, 20), image_zoom=0.5):
    """Plot images on a 2D projection (like UMAP or t-SNE)"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # First scatter the points to see the overall distribution
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
    
    # Then plot small versions of each image at its 2D location
    for i, (x, y) in enumerate(embeddings_2d):
        img = images[i]
        # Resize image for better visualization
        width, height = img.size
        new_width = int(width * image_zoom)
        new_height = int(height * image_zoom)
        img = img.resize((new_width, new_height))
        
        # Convert PIL image to a format matplotlib can use
        img_box = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(img_box, (x, y), frameon=True, pad=0.1)
        ax.add_artist(ab)
    
    # Set axis limits a bit beyond the data limits to see all images
    ax.set_xlim(embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1)
    ax.set_ylim(embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1)
    
    plt.title("UMAP Projection of Image Embeddings")
    plt.tight_layout()
    return fig, ax

# Visualize the UMAP embedding
print("Visualizing UMAP projection with images...")
fig, ax = plot_images_on_umap(umap_embeddings, large_images)
plt.show()

# %% [markdown]
# ### Using K-means Clustering on Embeddings
# 
# Now that we've visualized our embeddings in 2D space, let's use K-means clustering to identify groups of semantically similar images.

# %%
# Apply K-means clustering on the original high-dimensional embeddings
n_clusters = 5  # Increase the number of clusters for a more nuanced analysis
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(large_embeddings)

# Visualize clustering results on the UMAP projection
plt.figure(figsize=(15, 12))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                     c=clusters, cmap='viridis', s=100, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title(f'UMAP Projection with K-means Clustering (k={n_clusters})')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualizing Images by Cluster
# 
# Let's visualize the actual images in each cluster to see what semantic groupings the model has identified.

# %%
# Display images by cluster
for cluster_id in range(n_clusters):
    # Get indices of images in this cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    n_images_in_cluster = len(cluster_indices)
    
    if n_images_in_cluster > 0:
        # Calculate grid layout dimensions
        grid_cols = min(5, n_images_in_cluster)
        grid_rows = (n_images_in_cluster + grid_cols - 1) // grid_cols
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
        plt.suptitle(f'Cluster {cluster_id+1}: {n_images_in_cluster} Images')
        
        # Flatten axes array for easy iteration
        if grid_rows == 1 and grid_cols == 1:
            axes = np.array([axes])
        elif grid_rows == 1 or grid_cols == 1:
            axes = axes.flatten()
            
        # Plot each image in the cluster
        for i, idx in enumerate(cluster_indices):
            if i < len(axes):
                row, col = i // grid_cols, i % grid_cols
                if grid_rows == 1 and grid_cols == 1:
                    ax = axes[0]
                elif grid_rows == 1 or grid_cols == 1:
                    ax = axes[i]
                else:
                    ax = axes[row, col]
                    
                ax.imshow(large_images[idx])
                ax.set_title(f"Image {idx+1}")
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images_in_cluster, grid_rows * grid_cols):
            row, col = i // grid_cols, i % grid_cols
            if grid_rows == 1 and grid_cols == 1:
                pass  # No unused subplots in a 1x1 grid
            elif grid_rows == 1 or grid_cols == 1:
                if i < len(axes):
                    axes[i].axis('off')
            else:
                if row < grid_rows and col < grid_cols:
                    axes[row, col].axis('off')
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.show()

# %% [markdown]
# ### Analysis of Semantic Clustering
# 
# The clusters formed above demonstrate how SigLIP 2's embeddings group images based on semantic content rather than just visual similarity. This type of semantic clustering is valuable for:
# 
# 1. **Content organization**: Automatically categorizing large collections of images
# 2. **Recommendation systems**: Finding semantically related content
# 3. **Anomaly detection**: Identifying images that don't fit expected semantic patterns
# 4. **Dataset exploration**: Understanding the distribution of semantic concepts
# 
# The UMAP visualization provides insight into how the high-dimensional embedding space is organized, while K-means clustering identifies discrete groups within this space. Together, they offer a powerful way to explore and understand the semantic relationships captured by SigLIP 2's image embeddings.

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored the concept of image embeddings and specifically delved into SigLIP 2, Google's advanced multilingual vision-language encoder. We've seen how image embeddings work, the technical evolution from CLIP to SigLIP to SigLIP 2, and the key capabilities that make SigLIP 2 stand out.
# 
# Through practical examples, we've demonstrated:
# 
# 1. How to perform zero-shot image classification
# 2. How to compute image-text similarity
# 3. How to visualize and cluster embeddings
# 4. How to extract image embeddings for downstream tasks
# 5. How to compute image-to-image similarity
# 6. How to build a simple image search engine
# 
# Image embeddings like those produced by SigLIP 2 are foundational to modern computer vision applications, enabling efficient search, classification, and multimodal understanding. As models continue to evolve, we can expect even more powerful and versatile embeddings that further bridge the gap between vision and language understanding.
# 
# The flexible architecture and variant options make SigLIP 2 adaptable to a wide range of applications, from resource-constrained edge devices to high-performance systems requiring maximum accuracy. By understanding these tradeoffs, you can select the most appropriate SigLIP 2 variant for your specific use case, whether you prioritize efficiency, accuracy, or specialized capabilities like document understanding.
# 
# The multilingual capabilities and enhanced training methodology of SigLIP 2 make it particularly valuable for building more inclusive and accurate AI systems that can understand visual content across different languages and cultures.

# %% [markdown]
# ## Conclusion: The Power and Versatility of Image Embeddings
# 
# In this notebook, we've explored the concept of image embeddings with a focus on SigLIP 2, Google's advanced multilingual vision-language encoder. We've seen how these sophisticated representations go far beyond simple vector spaces, incorporating advanced mechanisms that significantly enhance their utility.
# 
# ### Key Takeaways
# 
# 1. **Advanced Similarity Computation**: SigLIP 2 doesn't just rely on simple cosine similarity between embeddings. It incorporates:
#    - MAP head pooling for better representation aggregation
#    - Temperature scaling to control similarity sharpness
#    - Bias terms to adjust for training imbalances
#    - Sigmoid activation to convert similarities to probabilities
# 
# 2. **Powerful Applications**: These sophisticated embeddings enable a wide range of applications:
#    - Visualization and exploration through clustering
#    - Unsupervised grouping based on semantic content
#    - Cross-modal understanding between images and text
#    - Semantic search engines with high precision
#    - Fine-grained recognition of subtle differences and similarities
# 
# 3. **Proper Usage**: As we've demonstrated, to get the most out of SigLIP 2, it's crucial to use the model's built-in similarity calculation mechanisms rather than trying to manually compute cosine similarity on raw embeddings.
# 
# The quality of SigLIP 2's embeddings makes these applications more accurate and robust than ever before. Its multilingual capabilities and improved semantic understanding make it particularly valuable for diverse global applications.
# 
# As image embedding models continue to evolve, we can expect even more powerful capabilities that further bridge the gap between visual content and natural language understanding. These embeddings form the foundation of modern computer vision systems and are becoming increasingly important in multimodal AI applications that combine vision, language, and other modalities.
# 
# Whether you're building a visual search engine, a content recommendation system, or a multimodal understanding application, image embeddings like those produced by SigLIP 2 provide a solid foundation for bringing semantic understanding to your visual data—just be sure to leverage their full capabilities by using the model's built-in similarity mechanisms!

# %% [markdown]
# ### Important Note on Processing Image-Text Pairs
# 
# An important detail when working with vision-language models like SigLIP is understanding how to properly compute similarity between images and text.
# 
# #### The Proper Way: Process Image-Text Pairs Together
# 
# While it's possible to extract image and text embeddings separately (as we did in some examples for educational purposes), the proper way to compute image-text similarity is to use the model's native capability to process image-text pairs together:
# 
# ```python
# # The right way to compute image-text similarity with vision-language models
# inputs = processor(text=texts, images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits_per_image  # Direct similarity scores
# probabilities = torch.sigmoid(logits)  # Convert to probabilities
# ```
# 
# #### Why This Matters
# 
# Vision-language models like SigLIP are specifically trained to compute similarity between image-text pairs in a particular way. When we extract embeddings separately and then compute similarity using dot products, we're not fully leveraging the model's capabilities.
# 
# The model's native `logits_per_image` output includes any internal transformations, normalization, or calibration that the model has learned during training. This leads to more accurate similarity scores compared to taking embeddings separately and computing similarity manually.
# 
# #### When to Use Direct Embeddings
# 
# There are still valid use cases for extracting embeddings directly:
# 
# 1. **Image-to-image similarity**: When comparing within the same modality
# 2. **Building search indices**: For efficient retrieval systems
# 3. **Transfer learning**: Using the embeddings as input features for downstream tasks
# 
# However, for direct image-text similarity comparisons, always prefer the model's built-in methods for processing the pairs together.


# %%
