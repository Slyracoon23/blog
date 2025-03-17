# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [raw]
# {
#   "aliases": ["/what-are-image-embeddings/"],
#   "categories": ["Computer Vision", "Machine Learning"],
#   "date": "2024-03-16",
#   "image": "/images/what_are_image_embeddings/thumbnail.jpg",
#   "title": "What are Image Embeddings?",
#   "subtitle": "Understanding how images are represented as numerical vectors for AI applications",
#   "format": "html"
# }

# %% [markdown]
# # What are Image Embeddings?
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
    """Extract embeddings from an image file or URL"""
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
    "a wild animal",
    "a train on tracks",
    "a person with an umbrella",
    "a child's toy",
    "a stop sign",
    "a bedroom",
    "a timepiece",
    "a vehicle for transportation"
]

# Get text embeddings using the processor and model
def get_text_embedding(text, model, processor):
    """Extract text embedding from a text string"""
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

# %%
# Compute similarity between our images and texts
similarity_matrix = np.zeros((len(images), len(texts)))
for i in range(len(images)):
    for j in range(len(texts)):
        similarity_matrix[i, j] = np.dot(embeddings[i], text_embeddings[j])

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
# Let's use clustering to group our images based on their semantic content.

# %%
# We'll need more images for meaningful clustering
# For demonstration, we'll use a smaller set but in practice you'd want more

# Let's use K-means clustering on our embeddings
if len(embeddings) >= 3:  # Only cluster if we have enough images
    n_clusters = min(3, len(embeddings))  # Use at most 3 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Display images by cluster
    for cluster_id in range(n_clusters):
        # Get indices of images in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        n_images_in_cluster = len(cluster_indices)
        
        if n_images_in_cluster > 0:
            # Display the images in this cluster
            fig, axes = plt.subplots(1, n_images_in_cluster, figsize=(15, 5))
            if n_images_in_cluster == 1:
                axes = [axes]  # Make it iterable when there's only one image
                
            plt.suptitle(f'Cluster {cluster_id+1} Images')
            
            for i, idx in enumerate(cluster_indices):
                axes[i].imshow(images[idx])
                axes[i].set_title(f"Image {idx+1}")
                axes[i].axis('off')
                
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
            plt.show()
else:
    print("Not enough images for clustering. Add more images for meaningful clusters.")

# %% [markdown]
# ### Example 4: Using SigLIP 2 for Image Embedding Extraction
# 
# One of the most common applications of models like SigLIP 2 is extracting image embeddings for downstream tasks such as clustering, similarity search, or fine-tuning classifiers. Let's see how to extract embeddings from sample images.

# %%
# We already did this in our get_image_embedding function
# Let's visualize the first 20 dimensions of each embedding

# Print shape of embeddings
print(f"Shape of image embeddings: {embeddings.shape}")

# Visualize the first 20 dimensions of each embedding
plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.plot(embeddings[i, :20], label=f"Image {i+1}")
plt.xlabel('Embedding Dimension')
plt.ylabel('Value')
plt.title('First 20 Dimensions of Image Embeddings')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Example 5: Computing Image-to-Image Similarity
# 
# Now that we have embeddings for our images, we can easily compute the similarity between them.

# %%
# Compute pairwise cosine similarity between images
image_sim = np.matmul(embeddings, embeddings.T)

# Visualize the similarity matrix
plt.figure(figsize=(8, 6))
plt.imshow(image_sim, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Cosine Similarity')
plt.xticks(np.arange(len(images)), [f"Image {i+1}" for i in range(len(images))])
plt.yticks(np.arange(len(images)), [f"Image {i+1}" for i in range(len(images))])
plt.title('Image-to-Image Similarity Matrix')
for i in range(len(images)):
    for j in range(len(images)):
        plt.text(j, i, f'{image_sim[i, j]:.2f}', 
                 ha='center', va='center', 
                 color='white' if abs(image_sim[i, j]) > 0.5 else 'black')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Example 6: Building a Simple Image Search Engine
# 
# Let's demonstrate how to build a simple image search engine using SigLIP 2 embeddings.

# %%
def search_similar_images(query_image_idx, embeddings, images, top_k=2):
    """Find the most similar images to a query image"""
    query_embedding = embeddings[query_image_idx]
    
    # Compute similarities
    similarities = np.dot(embeddings, query_embedding)
    
    # Sort by similarity (excluding the query image itself)
    similarities[query_image_idx] = -1  # Exclude the query image
    most_similar_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Show query image
    plt.subplot(1, top_k+1, 1)
    plt.imshow(images[query_image_idx])
    plt.title(f"Query Image {query_image_idx+1}")
    plt.axis('off')
    
    # Show similar images
    for i, idx in enumerate(most_similar_indices):
        plt.subplot(1, top_k+1, i+2)
        plt.imshow(images[idx])
        plt.title(f"Similar {i+1}: Image {idx+1}\nSimilarity: {similarities[idx]:.4f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Search for similar images using different query images
if len(images) >= 3:
    search_similar_images(0, embeddings, images, top_k=2)  # Using the first image as query
else:
    print("Not enough images for meaningful similarity search. Add more images for better results.")

# %% [markdown]
# ## Guidelines for Choosing the Right SigLIP 2 Variant
# 
# When selecting a SigLIP 2 variant for your application, consider these guidelines:
# 
# ### For Resource-Constrained Environments:
# - **Use Base models**: The ViT-B (86M) models provide a good balance between performance and efficiency
# - **Stick with 224px/256px resolution**: This minimizes memory usage and computation
# - **Consider quantization**: SigLIP 2 models can be quantized for further efficiency
# 
# ### For Production Applications with Balanced Requirements:
# - **SigLIP 2 So400m variants**: These shape-optimized models provide excellent accuracy without the computational demands of the largest models
# - **384px resolution**: A good balance between detail capture and resource usage
# - **Standard (non-NaFlex) variants**: Unless you specifically need aspect ratio preservation
# 
# ### For Maximum Quality:
# - **SigLIP 2 Giant (g) models**: These 1B parameter models provide the highest accuracy
# - **512px resolution**: Maximizes detail capture for complex scenes
# - **NaFlex variant**: If working with documents or aspect-sensitive content
# 
# ### For Document Understanding or OCR:
# - **NaFlex variants**: Essential for preserving text layout and aspect ratios
# - **Higher resolutions**: Crucial for legibility of small text
# - **Consider So400m or Large models**: These capture more fine-grained details

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

