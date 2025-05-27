# %% [markdown]
"""
# 🔥 PyTorch Fundamentals - Zero to Mastery
## Learn PyTorch for Deep Learning

### 📚 What You'll Learn in This Notebook:
- **Tensors**: The fundamental building blocks of machine learning
- **Creating Tensors**: Random, zeros, ones, ranges, and custom tensors
- **Tensor Operations**: Addition, multiplication, matrix multiplication
- **Shape Manipulation**: Reshaping, stacking, squeezing, unsqueezing
- **GPU Operations**: Moving tensors to GPU for faster computation
- **NumPy Integration**: Converting between PyTorch and NumPy
- **Reproducibility**: Using random seeds for consistent results

### 🎯 Learning Objectives:
By the end of this notebook, you'll understand:
1. What tensors are and why they're important
2. How to create and manipulate tensors
3. Matrix multiplication rules and applications
4. How to use GPUs for faster computation
5. Best practices for reproducible experiments

### 📖 Resources:
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Zero to Mastery PyTorch Course](https://github.com/mrdbourke/pytorch-deep-learning)
"""

# %% [markdown]
"""
## 🚀 Setup: Importing PyTorch and Checking Version

First, let's import PyTorch and check what version we're using.
PyTorch 1.10.0+ is recommended for this course.
"""

# %%
import torch
print(f"PyTorch version: {torch.__version__}")

# Check if we have access to GPU
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU (MPS) available")
else:
    print("Using CPU only")

# %% [markdown]
"""
# 🧮 Introduction to Tensors

## What are Tensors?
Tensors are the fundamental building block of machine learning and deep learning.
Their job is to represent data in a numerical way.

## Why Tensors Matter:
- **Images**: Can be represented as tensors with shape [height, width, color_channels]
- **Text**: Words can be converted to numbers and stored in tensors
- **Audio**: Sound waves can be digitized into tensor format
- **Any Data**: Almost any type of data can be represented as tensors

## Tensor Hierarchy:
- **Scalar** (0D): A single number
- **Vector** (1D): An array of numbers  
- **Matrix** (2D): A 2D array of numbers
- **Tensor** (3D+): An n-dimensional array of numbers

## Example Use Cases:
- 🖼️ **Image**: `[3, 224, 224]` → [color_channels, height, width]
- 📝 **Text**: `[batch_size, sequence_length]` → sentences as numbers
- 🎵 **Audio**: `[batch_size, time_steps, features]` → sound data
- 📊 **Tabular**: `[rows, columns]` → spreadsheet data
"""

# %% [markdown]
"""
## 📊 Creating Tensors: From Scalars to Multi-dimensional Arrays

Let's start by creating different types of tensors manually.
We'll progress from simple (scalar) to complex (multi-dimensional).
"""

# %% [markdown]
"""
### 🔢 Scalar (0 Dimensions)
A scalar is a single number. In tensor-speak, it's a zero-dimensional tensor.

**Use Cases**: 
- Single values like loss, accuracy, temperature
- Mathematical constants
- Single predictions
"""

# %%
scalar = torch.tensor(7)
print(f"Scalar: {scalar}")
print(f"Number of dimensions: {scalar.ndim}")
print(f"Shape: {scalar.shape}")
print(f"Get Python number from tensor: {scalar.item()}")
print(f"Data type: {scalar.dtype}")

# %% [markdown]
"""
### ➡️ Vector (1 Dimension)  
A vector is a single dimension tensor but can contain many numbers.

**Use Cases**:
- House features: [bedrooms, bathrooms, square_feet]
- Word embeddings: [0.2, -0.1, 0.9, ...]
- Time series: [price_day1, price_day2, price_day3, ...]
"""

# %%
vector = torch.tensor([7, 7])
print(f"Vector: {vector}")
print(f"Number of dimensions: {vector.ndim}")
print(f"Shape: {vector.shape}")
print(f"Size: {vector.size()}")

# Example: House features
house_features = torch.tensor([3, 2, 1500])  # [bedrooms, bathrooms, sqft]
print(f"\nHouse features example: {house_features}")
print(f"Bedrooms: {house_features[0]}, Bathrooms: {house_features[1]}, SqFt: {house_features[2]}")

# %% [markdown]
"""
### 🔲 Matrix (2 Dimensions)
A matrix is a 2D array of numbers. Very common in machine learning!

**Use Cases**:
- Images (grayscale): [height, width]  
- Spreadsheet data: [rows, columns]
- Neural network weights: [input_features, output_features]
- Batch of vectors: [batch_size, features]
"""

# %%
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(f"Matrix: {MATRIX}")
print(f"Number of dimensions: {MATRIX.ndim}")
print(f"Shape: {MATRIX.shape}")

# Example: Sales data
sales_data = torch.tensor([[100, 150, 200],    # Week 1: [Mon, Tue, Wed]
                          [120, 180, 220],    # Week 2: [Mon, Tue, Wed]  
                          [110, 160, 210]])   # Week 3: [Mon, Tue, Wed]
print(f"\nSales data example:")
print(f"Shape: {sales_data.shape} → [weeks, days]")
print(f"Sales data:\n{sales_data}")
print(f"Week 1 sales: {sales_data[0]}")
print(f"Monday sales across weeks: {sales_data[:, 0]}")

# %% [markdown]
"""
### 🧊 Tensor (3+ Dimensions)
Tensors with 3 or more dimensions. This is where things get interesting!

**Use Cases**:
- Color images: [height, width, color_channels] or [color_channels, height, width]
- Video data: [time, height, width, color_channels]
- Batch of images: [batch_size, color_channels, height, width]
- Text sequences: [batch_size, sequence_length, embedding_dim]
"""

# %%
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(f"Tensor: {TENSOR}")
print(f"Number of dimensions: {TENSOR.ndim}")
print(f"Shape: {TENSOR.shape}")

# Example: Mini RGB image (1x3x3 pixels)
mini_image = torch.tensor([[[255, 0, 0],      # Red channel
                           [0, 255, 0],      # Green channel  
                           [0, 0, 255]]])    # Blue channel
print(f"\nMini RGB image example:")
print(f"Shape: {mini_image.shape} → [color_channels, height, width]")
print(f"Red channel:\n{mini_image[0]}")

# %% [markdown]
"""
### 📏 Tensor Dimension Summary

| Name | Dimensions | Shape Example | Use Case |
|------|------------|---------------|----------|
| Scalar | 0 | `()` | Single values |
| Vector | 1 | `(5,)` | Features, embeddings |
| Matrix | 2 | `(3, 4)` | Images, spreadsheets |
| Tensor | 3+ | `(2, 3, 4)` | Videos, batches |

**Memory Tip**: Count the square brackets `[` on one side to get dimensions!
- `7` → 0 brackets → 0D (scalar)
- `[7, 8]` → 1 bracket → 1D (vector)  
- `[[7, 8], [9, 10]]` → 2 brackets → 2D (matrix)
- `[[[1, 2], [3, 4]]]` → 3 brackets → 3D (tensor)
"""

# %% [markdown]
"""
# 🎲 Random Tensors: The Starting Point of Machine Learning

## Why Random Tensors?
Machine learning models often start with large random tensors of numbers
and adjust these random numbers as they work through data to better represent patterns.

## The Learning Process:
1. **Start** with random numbers (poor representations)
2. **Look** at data 
3. **Update** random numbers to better represent data
4. **Repeat** steps 2-3 thousands of times
5. **Result**: Numbers that capture patterns in data!

## Common Random Tensor Functions:
- `torch.rand()`: Random numbers between 0 and 1 (uniform distribution)
- `torch.randn()`: Random numbers from normal distribution (mean=0, std=1)
- `torch.randint()`: Random integers within a range
"""

# %% [markdown]
"""
### 🎯 Creating Random Tensors with torch.rand()

`torch.rand()` creates tensors with random values between 0 and 1.
Perfect for initializing neural network weights!
"""

# %%
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(f"Random tensor (3, 4):")
print(f"Shape: {random_tensor.shape}")
print(f"Datatype: {random_tensor.dtype}")
print(f"Values between 0 and 1:\n{random_tensor}")

# Show min and max values
print(f"\nValue range check:")
print(f"Minimum value: {random_tensor.min()}")
print(f"Maximum value: {random_tensor.max()}")

# %% [markdown]
"""
### 🖼️ Real-World Example: Image-Sized Tensors

Images are commonly represented as tensors. Let's create a tensor 
in the shape of a typical image used in computer vision.

**Common Image Formats**:
- **PyTorch**: `[color_channels, height, width]` 
- **TensorFlow**: `[height, width, color_channels]`
- **Standard sizes**: 224x224, 256x256, 512x512 pixels
"""

# %%
# Create a random tensor in common image shape [height, width, color_channels]
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(f"Random image tensor:")
print(f"Shape: {random_image_size_tensor.shape} → [height, width, color_channels]")
print(f"Number of dimensions: {random_image_size_tensor.ndim}")
print(f"Total elements: {random_image_size_tensor.numel():,}")
print(f"Memory usage: ~{random_image_size_tensor.numel() * 4 / 1024:.1f} KB")

# PyTorch format (channels first)
pytorch_image = torch.rand(size=(3, 224, 224))
print(f"\nPyTorch format [C, H, W]: {pytorch_image.shape}")

# Batch of images (common in training)
batch_images = torch.rand(size=(32, 3, 224, 224))
print(f"Batch of images [batch, channels, height, width]: {batch_images.shape}")

# %% [markdown]
"""
### 🎲 Other Random Tensor Types

Different types of random distributions for different use cases.
"""

# %%
# Normal distribution (Gaussian) - common for weight initialization
normal_tensor = torch.randn(size=(3, 3))
print(f"Normal distribution tensor (mean≈0, std≈1):")
print(f"Tensor:\n{normal_tensor}")
print(f"Mean: {normal_tensor.mean():.3f}")
print(f"Standard deviation: {normal_tensor.std():.3f}")

# Random integers - useful for indices, labels
int_tensor = torch.randint(low=0, high=10, size=(3, 3))
print(f"\nRandom integers (0-9):")
print(f"Tensor:\n{int_tensor}")
print(f"Datatype: {int_tensor.dtype}")

# Random permutation - useful for shuffling data
perm_tensor = torch.randperm(10)
print(f"\nRandom permutation of 0-9: {perm_tensor}")
print("Useful for: shuffling datasets, random sampling")

# %% [markdown]
"""
## 🔢 Zeros and Ones Tensors

Sometimes you'll want to fill tensors with zeros or ones.
This is useful for:
- **Masking**: Hide certain values from the model
- **Initialization**: Start with known values
- **Placeholders**: Reserve space for data
"""


# %%
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(f"Zeros tensor: {zeros}")
print(f"Datatype: {zeros.dtype}")

# %%
# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(f"Ones tensor: {ones}")
print(f"Datatype: {ones.dtype}")

# %% [markdown]
"""
## 📊 Creating Ranges and Similar Tensors

Sometimes you need sequences of numbers or tensors with the same shape as existing ones.
"""


# %%
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(f"Range tensor: {zero_to_ten}")

# %%
# Create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(f"Zeros like tensor: {ten_zeros}")

# %% [markdown]
"""
## 🏷️ Tensor Datatypes

Different datatypes for different precision and speed requirements.

**Common Datatypes**:
- `torch.float32` (default): 32-bit floating point
- `torch.float16`: 16-bit floating point (faster, less precise)
- `torch.int64`: 64-bit integers
- `torch.bool`: Boolean values
"""

# %%
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # defaults to torch.float32
                               device=None,  # defaults to CPU
                               requires_grad=False)  # if True, operations are recorded

print(f"Float32 tensor: {float_32_tensor}")
print(f"Shape: {float_32_tensor.shape}")
print(f"Datatype: {float_32_tensor.dtype}")
print(f"Device: {float_32_tensor.device}")

# %%
# Create a float16 tensor
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(f"Float16 tensor datatype: {float_16_tensor.dtype}")

# %% [markdown]
"""
## 📋 Getting Information from Tensors

Three most common attributes you'll need to know about tensors:
- **shape**: What shape is the tensor?
- **dtype**: What datatype are the elements?
- **device**: What device is the tensor stored on? (CPU/GPU)

**Remember the debugging song**: "what shape, what datatype, where where where" 🎵
"""


# %%
# Create a tensor and find out details about it
some_tensor = torch.rand(3, 4)
print(f"Tensor: {some_tensor}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

# %% [markdown]
"""
# 🔧 Manipulating Tensors (Tensor Operations)

In deep learning, data gets represented as tensors and models learn by performing operations on tensors.

## Basic Operations:
- **Addition** (+)
- **Subtraction** (-)
- **Multiplication** (*) - element-wise
- **Division** (/)
- **Matrix multiplication** (@) - the most important!

These are the building blocks of neural networks! 🧱
"""

# %%
# Basic operations
tensor = torch.tensor([1, 2, 3])
print(f"Original tensor: {tensor}")
print(f"Add 10: {tensor + 10}")
print(f"Multiply by 10: {tensor * 10}")
print(f"Subtract 10: {tensor - 10}")

# %%
# Element-wise multiplication vs Matrix multiplication
print(f"Element-wise multiplication: {tensor} * {tensor} = {tensor * tensor}")
print(f"Matrix multiplication: {tensor} @ {tensor} = {torch.matmul(tensor, tensor)}")

# %% [markdown]
"""
## ⚡ Matrix Multiplication Rules

Matrix multiplication is the heart of deep learning! Here are the key rules:

1. **Inner dimensions must match**: 
   - `(3,2) @ (2,3)` ✅ Works!
   - `(3,2) @ (3,2)` ❌ Won't work!

2. **Resulting matrix has shape of outer dimensions**: 
   - `(3,2) @ (2,3)` → `(3,3)`
   - `(2,3) @ (3,2)` → `(2,2)`

**Memory tip**: Think of it as the inner dimensions "cancel out"
"""

# %%
# Matrix multiplication example
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

print(f"Tensor A shape: {tensor_A.shape}")
print(f"Tensor B shape: {tensor_B.shape}")

# This will work after transpose
print(f"Tensor B transposed shape: {tensor_B.T.shape}")
result = torch.matmul(tensor_A, tensor_B.T)
print(f"Matrix multiplication result: {result}")
print(f"Result shape: {result.shape}")

# %%
# Neural network linear layer example
torch.manual_seed(42)
linear = torch.nn.Linear(in_features=2, out_features=6)
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")

# %% [markdown]
"""
## 📊 Tensor Aggregation Operations

Aggregation operations reduce tensors from many values to fewer values.

**Common aggregations**:
- `tensor.min()` - minimum value
- `tensor.max()` - maximum value  
- `tensor.mean()` - average value
- `tensor.sum()` - sum of all values
- `tensor.argmin()` - index of minimum value
- `tensor.argmax()` - index of maximum value
"""

# %%
# Create a tensor and perform aggregation
x = torch.arange(0, 100, 10)
print(f"Tensor: {x}")
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.type(torch.float32).mean()}")  # requires float datatype
print(f"Sum: {x.sum()}")

# %%
# Positional min/max (argmin/argmax)
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

# %% [markdown]
"""
## 🔄 Changing Tensor Datatypes

Sometimes you need to convert tensors between different datatypes for compatibility.

**Common conversions**:
- `.type(torch.float32)` - Convert to 32-bit float
- `.type(torch.float16)` - Convert to 16-bit float (faster, less precise)
- `.type(torch.int8)` - Convert to 8-bit integer
"""


# %%
# Create a tensor and change its datatype
tensor = torch.arange(10., 100., 10.)
print(f"Original datatype: {tensor.dtype}")

# Create float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(f"Float16 tensor: {tensor_float16}")

# Create int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(f"Int8 tensor: {tensor_int8}")

# %% [markdown]
"""
# 🔧 Reshaping, Stacking, Squeezing and Unsqueezing

Shape manipulation is crucial in deep learning! Here are the key operations:

- **reshape()**: Change tensor shape (if compatible)
- **view()**: Create a new view of tensor (shares same data)
- **stack()**: Combine tensors along a new dimension
- **squeeze()**: Remove dimensions of size 1
- **unsqueeze()**: Add dimensions of size 1
- **permute()**: Rearrange dimensions
"""

# %%
# Create a tensor for reshaping examples
x = torch.arange(1., 8.)
print(f"Original tensor: {x}")
print(f"Original shape: {x.shape}")

# %%
# Reshape - add an extra dimension
x_reshaped = x.reshape(1, 7)
print(f"Reshaped tensor: {x_reshaped}")
print(f"Reshaped shape: {x_reshaped.shape}")

# %%
# View - creates a new view of the same tensor
z = x.view(1, 7)
print(f"View tensor: {z}")
print(f"View shape: {z.shape}")

# Changing view changes original tensor
z[:, 0] = 5
print(f"After changing view: {z}")
print(f"Original tensor also changed: {x}")

# %%
# Stack tensors
x_stacked = torch.stack([x, x, x, x], dim=0)
print(f"Stacked tensor: {x_stacked}")
print(f"Stacked shape: {x_stacked.shape}")

# %%
# Squeeze - remove dimensions with value 1
print(f"Before squeeze: {x_reshaped}")
print(f"Before squeeze shape: {x_reshaped.shape}")

x_squeezed = x_reshaped.squeeze()
print(f"After squeeze: {x_squeezed}")
print(f"After squeeze shape: {x_squeezed.shape}")

# %%
# Unsqueeze - add a dimension with value 1
print(f"Before unsqueeze: {x_squeezed}")
print(f"Before unsqueeze shape: {x_squeezed.shape}")

x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"After unsqueeze: {x_unsqueezed}")
print(f"After unsqueeze shape: {x_unsqueezed.shape}")

# %%
# Permute - rearrange dimensions
x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)  # shifts axis 0->1, 1->2, 2->0

print(f"Original shape: {x_original.shape}")
print(f"Permuted shape: {x_permuted.shape}")

# %% [markdown]
"""
## 🎯 Indexing (Selecting Data from Tensors)

Indexing allows you to select specific elements from tensors.

**Key concepts**:
- Use `[]` brackets to index
- `:` means "all values in this dimension"
- `,` separates different dimensions
- Indexing goes from outer → inner dimensions

**Examples**:
- `tensor[0]` - First element of first dimension
- `tensor[:, 1]` - All of first dim, index 1 of second dim
- `tensor[0, :, 2]` - Index 0 of first, all of second, index 2 of third
"""

# %%
# Create a tensor for indexing
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"Tensor: {x}")
print(f"Shape: {x.shape}")

# %%
# Index bracket by bracket
print(f"First square bracket: {x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket: {x[0][0][0]}")

# %%
# Using : and , for indexing
print(f"Get all values of 0th dimension and 0 index of 1st dimension: {x[:, 0]}")
print(f"Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension: {x[:, :, 1]}")
print(f"Get all values of 0 dimension but only 1 index of 1st and 2nd dimension: {x[:, 1, 1]}")
print(f"Get index 0 of 0th and 1st dimension and all values of 2nd dimension: {x[0, 0, :]}")

# %% [markdown]
"""
## 🔄 PyTorch Tensors & NumPy Integration

PyTorch plays nicely with NumPy! Here's how to convert between them:

**Key functions**:
- `torch.from_numpy(array)` - NumPy array → PyTorch tensor
- `tensor.numpy()` - PyTorch tensor → NumPy array

**Important notes**:
- NumPy arrays are usually `float64`, PyTorch defaults to `float32`
- Conversions create copies (changing one won't affect the other)
- GPU tensors must be moved to CPU before converting to NumPy
"""


# %%
import numpy as np

# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(f"NumPy array: {array}")
print(f"PyTorch tensor: {tensor}")
print(f"Tensor datatype: {tensor.dtype}")

# %%
# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(f"PyTorch tensor: {tensor}")
print(f"NumPy array: {numpy_tensor}")

# %% [markdown]
"""
## 🎲 Reproducibility: Taking the Random out of Random

Reproducibility is crucial for scientific experiments and debugging!

**Why use random seeds?**
- Get the same "random" numbers every time
- Make experiments reproducible
- Debug models consistently
- Share results that others can replicate

**Key functions**:
- `torch.manual_seed(42)` - Set CPU random seed
- `torch.cuda.manual_seed(42)` - Set GPU random seed
"""

# %%
# Create random tensors without seed (different each time)
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A: {random_tensor_A}")
print(f"Tensor B: {random_tensor_B}")
print(f"Are they equal? {torch.equal(random_tensor_A, random_tensor_B)}")

# %%
# Create random tensors with seed (same each time)
RANDOM_SEED = 42

torch.manual_seed(seed=RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(seed=RANDOM_SEED)  # Reset seed for same result
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C: {random_tensor_C}")
print(f"Tensor D: {random_tensor_D}")
print(f"Are they equal? {torch.equal(random_tensor_C, random_tensor_D)}")

# %% [markdown]
"""
# ⚡ Running Tensors on GPUs

GPUs can make computations much faster for deep learning!

**Why use GPUs?**
- **Speed**: 10-100x faster than CPUs for deep learning
- **Parallel processing**: Handle thousands of operations simultaneously
- **Memory**: Large amounts of fast memory for big models

**Device types**:
- **CPU**: Always available, slower for ML
- **CUDA**: NVIDIA GPUs, most common for deep learning
- **MPS**: Apple Silicon GPUs (M1/M2/M3)

**Key functions**:
- `torch.cuda.is_available()` - Check for NVIDIA GPU
- `torch.backends.mps.is_available()` - Check for Apple Silicon
- `tensor.to(device)` - Move tensor to device
"""


# %%
# Check for GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")

# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# %%
# Count number of GPUs
if torch.cuda.is_available():
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# %%
# Create tensor and move to GPU
tensor = torch.tensor([1, 2, 3])
print(f"Tensor on CPU: {tensor}, device: {tensor.device}")

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(f"Tensor on GPU: {tensor_on_gpu}")

# %%
# Moving tensors back to CPU (needed for NumPy operations)
if device != "cpu":
    tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
    print(f"Tensor back on CPU as NumPy: {tensor_back_on_cpu}")
    print(f"Original tensor still on GPU: {tensor_on_gpu}")

# %%
"""
# 🏋️‍♂️ EXERCISES - PyTorch Fundamentals Practice

All exercises are focused on practicing the code concepts covered above.
You should be able to complete them by referencing each section.

## Resources:
- PyTorch Documentation: https://pytorch.org/docs/stable/
- torch.Tensor documentation: https://pytorch.org/docs/stable/tensors.html
- torch.cuda documentation: https://pytorch.org/docs/stable/cuda.html

## Exercise List:
1. Create a random tensor with shape (7, 7)
2. Perform matrix multiplication on tensor from #1 with another random tensor with shape (1, 7)
3. Set random seed to 0 and repeat exercises 1 & 2
4. Set GPU random seed (if available)
5. Create two random tensors of shape (2, 3) and send to GPU
6. Perform matrix multiplication on tensors from #5
7. Find max and min values of output from #6
8. Find max and min index values of output from #6
9. Make random tensor with shape (1, 1, 1, 10) and remove all 1 dimensions
"""

# %% [markdown]
"""
## Exercise 1: Create a random tensor with shape (7, 7)

**Goal**: Practice creating random tensors with specific shapes
**Concepts**: torch.rand(), tensor shapes, random number generation
"""
torch.manual_seed(42)  # For reproducibility
exercise_tensor = torch.rand(7, 7)
print(f"Exercise 1 - Random tensor (7, 7):")
print(f"Shape: {exercise_tensor.shape}")
print(f"Tensor:\\n{exercise_tensor}")
print(f"Datatype: {exercise_tensor.dtype}")

# %% [markdown]
"""
## Exercise 2: Matrix multiplication with different shaped tensors

**Goal**: Practice matrix multiplication and understand shape requirements
**Concepts**: torch.matmul(), tensor shapes, transpose operations
**Key Learning**: Inner dimensions must match for matrix multiplication
"""
torch.manual_seed(42)
tensor1 = torch.rand(7, 7)  # Shape: (7, 7)
tensor2 = torch.rand(1, 7)  # Shape: (1, 7)

print(f"Tensor 1 shape: {tensor1.shape}")
print(f"Tensor 2 shape: {tensor2.shape}")
print(f"Tensor 2 transposed shape: {tensor2.T.shape}")

# Matrix multiplication: (7, 7) @ (7, 1) = (7, 1)
result = torch.matmul(tensor1, tensor2.T)
print(f"\nExercise 2 - Matrix multiplication result:")
print(f"Result shape: {result.shape}")
print(f"Result:\\n{result}")

# %% [markdown]
"""
## Exercise 3: Reproducibility with random seeds

**Goal**: Understand how random seeds affect reproducibility
**Concepts**: torch.manual_seed(), reproducible experiments
**Key Learning**: Same seed = same "random" numbers
"""
print("=== Setting seed to 0 and repeating exercises ===")

# Exercise 1 with seed 0
torch.manual_seed(0)
tensor1_seed0 = torch.rand(7, 7)
print(f"Exercise 1 with seed 0 - Tensor shape: {tensor1_seed0.shape}")

# Exercise 2 with seed 0  
torch.manual_seed(0)
tensor1_seed0 = torch.rand(7, 7)
tensor2_seed0 = torch.rand(1, 7)
result_seed0 = torch.matmul(tensor1_seed0, tensor2_seed0.T)
print(f"Exercise 2 with seed 0 - Result shape: {result_seed0.shape}")
print(f"Result:\\n{result_seed0}")

# Demonstrate reproducibility
torch.manual_seed(0)
tensor_reproducible_1 = torch.rand(3, 3)
torch.manual_seed(0)
tensor_reproducible_2 = torch.rand(3, 3)
print(f"\nReproducibility test - Are tensors equal? {torch.equal(tensor_reproducible_1, tensor_reproducible_2)}")

# %% [markdown]
"""
## Exercise 4: GPU random seed

**Goal**: Learn about GPU-specific random seeds
**Concepts**: torch.cuda.manual_seed(), GPU vs CPU random states
**Key Learning**: GPU and CPU have separate random number generators
"""
print("=== GPU Random Seed Setup ===")

if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
    print("✅ GPU (CUDA) random seed set to 1234")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Demonstrate GPU random seed
    torch.cuda.manual_seed(1234)
    gpu_tensor_1 = torch.rand(2, 2, device='cuda')
    torch.cuda.manual_seed(1234)
    gpu_tensor_2 = torch.rand(2, 2, device='cuda')
    print(f"GPU tensors equal with same seed: {torch.equal(gpu_tensor_1, gpu_tensor_2)}")
    
elif torch.backends.mps.is_available():
    # Note: MPS doesn't have a separate manual_seed function
    print("✅ Apple Silicon (MPS) detected")
    print("Note: MPS uses the same random seed as CPU (torch.manual_seed)")
    
else:
    print("❌ No GPU available - using CPU only")
    print("GPU random seed not applicable")

# %% [markdown]
"""
## Exercise 5: Create tensors and move to GPU

**Goal**: Practice moving tensors between devices (CPU ↔ GPU)
**Concepts**: .to(device), device-agnostic code, GPU memory management
**Key Learning**: Tensors can be moved between devices for faster computation
"""
print("=== Creating tensors and moving to GPU ===")

torch.manual_seed(1234)
tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(2, 3)

print(f"Original tensors on CPU:")
print(f"Tensor A: {tensor_a}")
print(f"Tensor B: {tensor_b}")
print(f"Tensor A device: {tensor_a.device}")

if device != "cpu":
    tensor_a_gpu = tensor_a.to(device)
    tensor_b_gpu = tensor_b.to(device)
    print(f"\n✅ Tensors moved to {device}:")
    print(f"Tensor A on GPU: {tensor_a_gpu}")
    print(f"Tensor B on GPU: {tensor_b_gpu}")
    print(f"Tensor A GPU device: {tensor_a_gpu.device}")
else:
    print("\n❌ No GPU available, tensors remain on CPU")
    tensor_a_gpu = tensor_a
    tensor_b_gpu = tensor_b

# %% [markdown]
"""
## Exercise 6: Matrix multiplication on GPU

**Goal**: Perform computations on GPU for faster processing
**Concepts**: GPU matrix multiplication, shape manipulation for matmul
**Key Learning**: GPU operations are faster but require compatible shapes
"""
print("=== Matrix multiplication on GPU ===")

print(f"Tensor A shape: {tensor_a_gpu.shape}")  # (2, 3)
print(f"Tensor B shape: {tensor_b_gpu.shape}")  # (2, 3)
print(f"Tensor B transposed shape: {tensor_b_gpu.T.shape}")  # (3, 2)

# Matrix multiplication: (2, 3) @ (3, 2) = (2, 2)
result_gpu = torch.matmul(tensor_a_gpu, tensor_b_gpu.T)

if device != "cpu":
    print(f"✅ GPU matrix multiplication completed")
    print(f"Result device: {result_gpu.device}")
else:
    print(f"✅ CPU matrix multiplication completed")

print(f"Result shape: {result_gpu.shape}")
print(f"Result:\\n{result_gpu}")

# %% [markdown]
"""
## Exercise 7: Find maximum and minimum values

**Goal**: Practice tensor aggregation operations
**Concepts**: .max(), .min(), tensor reduction operations
**Key Learning**: Aggregation reduces tensor dimensions to single values
"""
print("=== Finding max and min values ===")

max_value = result_gpu.max()
min_value = result_gpu.min()

print(f"Original tensor:\\n{result_gpu}")
print(f"📊 Aggregation results:")
print(f"Maximum value: {max_value}")
print(f"Minimum value: {min_value}")
print(f"Max value type: {type(max_value)}")

# Additional aggregations for learning
mean_value = result_gpu.mean()
sum_value = result_gpu.sum()
print(f"\n📈 Additional aggregations:")
print(f"Mean value: {mean_value}")
print(f"Sum value: {sum_value}")

# %% [markdown]
"""
## Exercise 8: Find maximum and minimum index positions

**Goal**: Learn about positional aggregation operations
**Concepts**: .argmax(), .argmin(), tensor indexing
**Key Learning**: argmax/argmin return positions, not values
"""
print("=== Finding max and min index positions ===")

max_index = result_gpu.argmax()
min_index = result_gpu.argmin()

print(f"Original tensor:\\n{result_gpu}")
print(f"Tensor flattened: {result_gpu.flatten()}")
print(f"\n🎯 Index positions:")
print(f"Maximum value index: {max_index}")
print(f"Minimum value index: {min_index}")

# Verify the indices are correct
flattened = result_gpu.flatten()
print(f"\n✅ Verification:")
print(f"Value at max index {max_index}: {flattened[max_index]}")
print(f"Value at min index {min_index}: {flattened[min_index]}")
print(f"Max value matches: {flattened[max_index] == result_gpu.max()}")
print(f"Min value matches: {flattened[min_index] == result_gpu.min()}")

# %% [markdown]
"""
## Exercise 9: Remove dimensions of size 1 (squeezing)

**Goal**: Practice tensor shape manipulation
**Concepts**: .squeeze(), tensor dimensions, shape transformation
**Key Learning**: squeeze() removes all dimensions with size 1
"""
print("=== Removing dimensions of size 1 ===")

torch.manual_seed(7)
tensor_with_ones = torch.rand(1, 1, 1, 10)

print(f"📏 Original tensor info:")
print(f"Shape: {tensor_with_ones.shape}")
print(f"Number of dimensions: {tensor_with_ones.ndim}")
print(f"Tensor:\\n{tensor_with_ones}")

# Remove all dimensions of size 1
tensor_squeezed = tensor_with_ones.squeeze()

print(f"\n📏 After squeezing:")
print(f"Shape: {tensor_squeezed.shape}")
print(f"Number of dimensions: {tensor_squeezed.ndim}")
print(f"Tensor: {tensor_squeezed}")

# Demonstrate selective squeezing
tensor_selective = tensor_with_ones.squeeze(0)  # Remove only first dimension
print(f"\n📏 Selective squeeze (dim=0):")
print(f"Shape: {tensor_selective.shape}")

# Show the reverse operation (unsqueeze)
tensor_unsqueezed = tensor_squeezed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
print(f"\n📏 After unsqueezing back:")
print(f"Shape: {tensor_unsqueezed.shape}")
print(f"Shapes match original: {tensor_unsqueezed.shape == tensor_with_ones.shape}")

# %% [markdown]
"""
# 🎯 Exercise Summary & Key Takeaways

## What You've Learned:
✅ **Tensor Creation**: Random tensors, shapes, and datatypes
✅ **Matrix Multiplication**: Shape requirements and operations
✅ **Reproducibility**: Using seeds for consistent results
✅ **GPU Operations**: Moving tensors and computations to GPU
✅ **Aggregation**: Finding min, max, mean, sum values
✅ **Indexing**: Finding positions of min/max values
✅ **Shape Manipulation**: Squeezing and unsqueezing dimensions

## Next Steps:
1. 📖 Read PyTorch documentation for 10 minutes
2. 🎥 Watch "What's a tensor?" video
3. 🔄 Try modifying the exercises with different shapes
4. 🧪 Experiment with different tensor operations
5. 📝 Practice writing device-agnostic code

## Common Patterns to Remember:
- Always check tensor shapes before operations
- Use `.to(device)` for device-agnostic code
- Set random seeds for reproducible experiments
- Use `.squeeze()` and `.unsqueeze()` for shape manipulation
- GPU operations are faster but require more memory management

## Debugging Tips:
- Shape errors? Check inner dimensions for matrix multiplication
- Device errors? Ensure all tensors are on the same device
- Datatype errors? Convert tensors to compatible types
- Remember: "what shape, what datatype, where where where" 🎵
"""

# %%
print("🎉 All exercises completed successfully!")
print("You're now ready for PyTorch Workflow Fundamentals!")

# %%
print("🎉 PyTorch Fundamentals Complete!")
print("You've learned about:")
print("- Creating tensors (scalars, vectors, matrices)")
print("- Random tensors and reproducibility")
print("- Tensor operations and matrix multiplication")
print("- Reshaping and indexing")
print("- GPU operations")
print("- NumPy integration")

# %%
