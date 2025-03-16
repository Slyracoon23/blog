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
# This notebook explores the concept of image embeddings, how they work, and their applications in AI. We'll focus on Google's SigLIP 2, a state-of-the-art multilingual vision-language encoder, and demonstrate its implementation.

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
# SigLIP 2 represents the latest advancement in image embedding technology. Developed by Google and released in early 2025, it significantly improves upon its predecessor by offering enhanced semantic understanding, better localization capabilities, and more effective dense feature representation.

# %% [markdown]
# ### Technical Background and Evolution
# 
# #### From CLIP to SigLIP to SigLIP 2
# 
# Vision-language models have evolved considerably in recent years:
# 
# 1. **CLIP and ALIGN**: These pioneered the approach of jointly training image and text encoders to understand the semantic relationship between visual data and natural language
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
# ## Implementing SigLIP 2: Practical Examples
# 
# Now that we understand the theoretical background of image embeddings and SigLIP 2, let's implement it to see how it works in practice. We'll use the Hugging Face Transformers library, which provides easy access to SigLIP 2 models.

# %%
# First, let's install the necessary libraries
!pip install torch transformers pillow requests matplotlib numpy

# %%
# Import the required libraries
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ### Loading the SigLIP 2 Model
# 
# We'll use the base variant of SigLIP 2 for our examples. The model we're loading is the multilingual version (i18n) with a ViT-Base backbone.

# %%
# Load the SigLIP 2 model and processor
model_name = "google/siglip-base-patch16-256-i18n"

try:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print(f"Successfully loaded {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using a placeholder for demonstration purposes.")
    # This is just for demonstration if the actual model can't be loaded
    class DummyModel:
        def __call__(self, **kwargs):
            class DummyOutput:
                def __init__(self):
                    batch_size = kwargs.get('pixel_values', torch.ones(1, 3, 224, 224)).shape[0]
                    text_batch_size = kwargs.get('input_ids', torch.ones(1, 10)).shape[0]
                    self.logits_per_image = torch.rand(batch_size, text_batch_size)
                    self.logits_per_text = self.logits_per_image.T
            return DummyOutput()
    
    class DummyProcessor:
        def __call__(self, text=None, images=None, return_tensors=None, padding=None, max_length=None):
            result = {}
            if images is not None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
                result['pixel_values'] = torch.ones(batch_size, 3, 224, 224)
            if text is not None:
                if isinstance(text, list):
                    batch_size = len(text)
                else:
                    batch_size = 1
                result['input_ids'] = torch.ones(batch_size, 10, dtype=torch.long)
                result['attention_mask'] = torch.ones(batch_size, 10, dtype=torch.long)
            return result
    
    processor = DummyProcessor()
    model = DummyModel()

# %% [markdown]
# ### Example 1: Zero-Shot Image Classification
# 
# Let's use SigLIP 2 for zero-shot image classification. We'll load an image and classify it against different text prompts.

# %%
# Download a sample image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title("Sample image")
plt.show()

# Define potential classes as text prompts
candidate_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person"
]

# Process the inputs
inputs = processor(text=candidate_labels, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = torch.sigmoid(logits_per_image)  # convert to probabilities

# Display the results
probs_np = probs.cpu().numpy()[0]
for label, prob in zip(candidate_labels, probs_np):
    print(f"{label}: {prob:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar([label.replace("a photo of a ", "") for label in candidate_labels], probs_np)
plt.xlabel('Category')
plt.ylabel('Probability')
plt.title('Zero-Shot Classification Results')
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# ### Example 2: Image-Text Similarity
# 
# Now let's explore how we can use SigLIP 2 to compute similarity between multiple images and texts.

# %%
# Download additional sample images
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
    "http://images.cocodataset.org/val2017/000000252219.jpg",  # dog
    "http://images.cocodataset.org/val2017/000000578967.jpg"   # person on bicycle
]

images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, img) in enumerate(zip(axes, images)):
    ax.imshow(img)
    ax.set_title(f"Image {i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Text descriptions
texts = [
    "two cats lying together",
    "a dog in the grass",
    "a person riding a bicycle",
    "a car on the street"
]

# %% [markdown]
# Let's compute the similarity between all these images and texts:

# %%
# Process all images
image_inputs = processor(images=images, return_tensors="pt")
image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

# Process all texts
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

# Compute embeddings
with torch.no_grad():
    # Get image embeddings
    image_embeds = model.get_image_features(**{k: v for k, v in image_inputs.items() 
                                             if k in ['pixel_values']})
    
    # Get text embeddings
    text_embeds = model.get_text_features(**{k: v for k, v in text_inputs.items()
                                           if k in ['input_ids', 'attention_mask']})
    
    # Normalize embeddings
    image_embeds = torch.nn.functional.normalize(image_embeds, dim=1)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=1)
    
    # Compute similarity
    similarity = torch.matmul(image_embeds, text_embeds.T)
    sigmoid_sim = torch.sigmoid(similarity)

# Display similarity matrix
similarity_np = sigmoid_sim.cpu().numpy()

plt.figure(figsize=(10, 8))
plt.imshow(similarity_np, vmin=0, vmax=1, cmap='viridis')
plt.colorbar(label='Similarity (sigmoid)')
plt.xticks(np.arange(len(texts)), texts, rotation=45, ha='right')
plt.yticks(np.arange(len(images)), [f"Image {i+1}" for i in range(len(images))])
plt.title('Image-Text Similarity Matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Example 3: Using SigLIP 2 for Image Embedding Extraction
# 
# One of the most common applications of models like SigLIP 2 is extracting image embeddings for downstream tasks such as clustering, similarity search, or fine-tuning classifiers. Let's see how to extract embeddings from our sample images.

# %%
# Extract embeddings from our sample images
with torch.no_grad():
    image_embeddings = model.get_image_features(**{k: v for k, v in image_inputs.items() 
                                                 if k in ['pixel_values']})
    
    # Normalize the embeddings
    image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)

# Convert to numpy for easier analysis
image_embeddings_np = image_embeddings.cpu().numpy()

# Print shape of embeddings
print(f"Shape of image embeddings: {image_embeddings_np.shape}")

# Visualize the first 20 dimensions of each embedding
plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.plot(image_embeddings_np[i, :20], label=f"Image {i+1}")
plt.xlabel('Embedding Dimension')
plt.ylabel('Value')
plt.title('First 20 Dimensions of Image Embeddings')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Example 4: Computing Image-to-Image Similarity
# 
# Now that we have embeddings for our images, we can easily compute the similarity between them.

# %%
# Compute pairwise cosine similarity between images
image_sim = torch.matmul(image_embeddings, image_embeddings.T).cpu().numpy()

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
# ## Conclusion
# 
# In this notebook, we've explored the concept of image embeddings and specifically delved into SigLIP 2, Google's advanced multilingual vision-language encoder. We've seen how image embeddings work, the technical evolution from CLIP to SigLIP to SigLIP 2, and the key capabilities that make SigLIP 2 stand out.
# 
# Through practical examples, we've demonstrated:
# 
# 1. How to perform zero-shot image classification
# 2. How to compute image-text similarity
# 3. How to extract image embeddings for downstream tasks
# 4. How to compute image-to-image similarity
# 
# Image embeddings like those produced by SigLIP 2 are foundational to modern computer vision applications, enabling efficient search, classification, and multimodal understanding. As models continue to evolve, we can expect even more powerful and versatile embeddings that further bridge the gap between vision and language understanding.
# 
# The multilingual capabilities and enhanced training methodology of SigLIP 2 make it particularly valuable for building more inclusive and accurate AI systems that can understand visual content across different languages and cultures.

