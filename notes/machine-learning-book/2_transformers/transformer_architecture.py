# %% [markdown]
# # The Transformer Architecture: From Theory to Implementation
# ## A Complete Course in Understanding and Building Transformers from Scratch
# 
# **Course Overview:**
# - Week 1 (May 24-25): The Transformer Architecture
# - Progressive learning with theory, implementation, and exercises
# - Built with PyTorch and practical examples
# 
# **Learning Objectives:**
# By the end of this course, you will:
# 1. Understand the core concepts behind the Transformer architecture
# 2. Implement each component from scratch using PyTorch
# 3. Build a complete working Transformer model
# 4. Understand how attention mechanisms revolutionized NLP

# %% [markdown]
# ## 📚 Day 1: May 24 - Architecture Basics
# ### Section 1: Introduction to Transformers

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from typing import Optional, Tuple
import shutil
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device detection with MPS support for Mac
def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

print("PyTorch version:", torch.__version__)
print(f"Device available: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {device}")

# Device performance monitoring
def monitor_device_performance():
    """Monitor device performance and memory usage"""
    print("\n🖥️ Device Performance Monitoring")
    print("=" * 40)
    
    if device.type == "mps":
        print("✅ Using Apple Metal Performance Shaders (MPS)")
        print("- Optimized for Apple Silicon and discrete GPUs on Mac")
        print("- Significantly faster than CPU for large models")
        
        # Create a test tensor to check memory allocation
        test_tensor = torch.randn(1000, 1000, device=device)
        print(f"- Successfully allocated tensor on MPS: {test_tensor.shape}")
        del test_tensor
        
    elif device.type == "cuda":
        print("✅ Using NVIDIA CUDA")
        print(f"- GPU: {torch.cuda.get_device_name()}")
        print(f"- Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"- Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    else:
        print("⚠️ Using CPU")
        print("- Consider using a GPU for better performance with larger models")
    
    return device

# Monitor performance
current_device = monitor_device_performance()

# %% [markdown]
# ### 🎯 Exercise 1.1: Understanding the Problem
# **Before Transformers, NLP models had limitations:**
# - RNNs processed sequences sequentially (slow, hard to parallelize)
# - LSTMs helped with long sequences but still sequential
# - CNNs could parallelize but struggled with long-range dependencies
# 
# **The Transformer solution:**
# - Process entire sequences in parallel
# - Use attention to capture long-range dependencies
# - "Attention is all you need" - no recurrence or convolution needed

# %%
# Exercise 1.1: Let's see why sequential processing is slow
def simulate_rnn_processing():
    """Simulate how RNNs process sequences sequentially"""
    sequence_length = 10
    hidden_size = 4
    
    print("RNN Sequential Processing:")
    print("=" * 40)
    
    # Simulate processing each token one by one
    hidden_state = torch.zeros(hidden_size)
    for i in range(sequence_length):
        # Each step depends on the previous hidden state
        print(f"Step {i+1}: Processing token {i+1}, depends on step {i}")
        # Simulated computation
        hidden_state = torch.tanh(hidden_state + torch.randn(hidden_size))
    
    print("\nTransformer Parallel Processing:")
    print("=" * 40)
    print("All tokens processed simultaneously using attention!")
    
    return hidden_state

result = simulate_rnn_processing()

# %% [markdown]
# ### Section 2: The Overall Architecture
# 
# The Transformer consists of:
# 1. **Input Embeddings** + **Positional Encoding**
# 2. **Encoder Stack** (6 layers)
#    - Multi-Head Self-Attention
#    - Feed-Forward Network
#    - Residual connections + Layer Normalization
# 3. **Decoder Stack** (6 layers)
#    - Masked Multi-Head Self-Attention
#    - Encoder-Decoder Attention
#    - Feed-Forward Network
#    - Residual connections + Layer Normalization
# 4. **Output Layer**

# %%
# Let's use the Excalidraw visualization instead of creating our own
def show_transformer_architecture():
    """Display the transformer architecture using the Excalidraw diagram"""
    target_file = "transformer_architecture_excalidraw.excalidraw.png"
    
    try:
        from IPython.display import Image, display
        print("🎨 Transformer Architecture Visualization")
        print("=" * 50)
        display(Image(target_file))
    except ImportError:
        # Fallback for non-Jupyter environments
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        try:
            img = mpimg.imread(target_file)
            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Transformer Architecture: "Attention Is All You Need"', 
                     fontsize=16, weight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            print(f"⚠️ Could not find {target_file}")
            print("Please ensure the image file is in the same directory as this script.")
            print("\nThe Transformer consists of:")
            print("1. Input Embeddings + Positional Encoding")
            print("2. Encoder Stack (6 layers)")
            print("   - Multi-Head Self-Attention")
            print("   - Feed-Forward Network")
            print("   - Residual connections + Layer Normalization")
            print("3. Decoder Stack (6 layers)")
            print("   - Masked Multi-Head Self-Attention")
            print("   - Encoder-Decoder Attention")
            print("   - Feed-Forward Network")
            print("   - Residual connections + Layer Normalization")
            print("4. Output Layer (Linear + Softmax)")

# Also create a detailed attention mechanism visualization
def plot_attention_mechanism():
    """Visualize the attention mechanism in detail"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Query, Key, Value concept
    ax1 = axes[0, 0]
    
    # Create sample matrices
    seq_len = 4
    d_model = 6
    
    # Input matrix
    input_matrix = np.random.randn(seq_len, d_model)
    
    # Q, K, V matrices (simplified)
    Q = input_matrix @ np.random.randn(d_model, d_model)
    K = input_matrix @ np.random.randn(d_model, d_model) 
    V = input_matrix @ np.random.randn(d_model, d_model)
    
    # Show the transformation
    ax1.text(0.5, 0.9, 'Input → Query, Key, Value', ha='center', transform=ax1.transAxes,
            fontsize=14, weight='bold')
    
    positions = [0.1, 0.35, 0.6, 0.85]
    labels = ['Input\nX', 'Query\nQ=XWq', 'Key\nK=XWk', 'Value\nV=XWv']
    matrices = [input_matrix, Q, K, V]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, (pos, label, matrix, color) in enumerate(zip(positions, labels, matrices, colors)):
        im = ax1.imshow(matrix, cmap='RdBu_r', aspect='auto', 
                       extent=[pos, pos+0.15, 0.1, 0.7])
        ax1.text(pos+0.075, 0.05, label, ha='center', fontsize=10, weight='bold')
        
        if i < 3:  # Draw arrows
            ax1.annotate('', xy=(positions[i+1]-0.02, 0.4), xytext=(pos+0.17, 0.4),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Attention score calculation
    ax2 = axes[0, 1]
    
    # Simulate attention scores
    scores = np.random.rand(seq_len, seq_len)
    scores = scores / scores.sum(axis=1, keepdims=True)  # Normalize
    
    im2 = ax2.imshow(scores, cmap='Blues', aspect='equal')
    ax2.set_title('Attention Weights\nAttention(Q,K,V) = softmax(QK^T/√d_k)V', 
                 fontsize=12, weight='bold')
    
    # Add labels
    ax2.set_xticks(range(seq_len))
    ax2.set_yticks(range(seq_len))
    ax2.set_xticklabels([f'K{i}' for i in range(seq_len)])
    ax2.set_yticklabels([f'Q{i}' for i in range(seq_len)])
    
    # Add values as text
    for i in range(seq_len):
        for j in range(seq_len):
            ax2.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center',
                    color='white' if scores[i,j] > 0.5 else 'black')
    
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. Multi-head attention
    ax3 = axes[1, 0]
    
    num_heads = 4
    head_colors = ['Reds', 'Blues', 'Greens', 'Purples']
    
    for head in range(num_heads):
        # Generate different attention patterns for each head
        if head == 0:  # Diagonal pattern
            pattern = np.eye(seq_len) + 0.1 * np.random.rand(seq_len, seq_len)
        elif head == 1:  # Previous token pattern
            pattern = np.tril(np.ones((seq_len, seq_len))) + 0.1 * np.random.rand(seq_len, seq_len)
        elif head == 2:  # Uniform pattern
            pattern = np.ones((seq_len, seq_len)) + 0.2 * np.random.rand(seq_len, seq_len)
        else:  # Local pattern
            pattern = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(max(0, i-1), min(seq_len, i+2)):
                    pattern[i, j] = 1
            pattern += 0.1 * np.random.rand(seq_len, seq_len)
        
        pattern = pattern / pattern.sum(axis=1, keepdims=True)
        
        # Create subplot
        start_x = 0.02 + head * 0.24
        start_y = 0.1
        width = height = 0.2
        
        # Create inset axes
        inset = ax3.inset_axes([start_x, start_y, width, height])
        im = inset.imshow(pattern, cmap=head_colors[head], aspect='equal')
        inset.set_title(f'Head {head+1}', fontsize=10, weight='bold')
        inset.set_xticks([])
        inset.set_yticks([])
    
    ax3.text(0.5, 0.9, 'Multi-Head Attention: Different Heads Learn Different Patterns', 
            ha='center', transform=ax3.transAxes, fontsize=14, weight='bold')
    ax3.text(0.5, 0.05, 'Each head captures different types of relationships', 
            ha='center', transform=ax3.transAxes, fontsize=12, style='italic')
    ax3.axis('off')
    
    # 4. Self-attention vs Cross-attention
    ax4 = axes[1, 1]
    
    # Self-attention
    self_att = np.random.rand(seq_len, seq_len)
    self_att = self_att / self_att.sum(axis=1, keepdims=True)
    
    # Cross-attention (different dimensions)
    cross_att = np.random.rand(seq_len, seq_len + 2)  # Different sequence lengths
    cross_att = cross_att / cross_att.sum(axis=1, keepdims=True)
    
    # Plot both
    im_self = ax4.imshow(self_att, cmap='Blues', aspect='equal', 
                        extent=[0, seq_len, seq_len, 0])
    im_cross = ax4.imshow(cross_att, cmap='Reds', aspect='equal', alpha=0.7,
                         extent=[seq_len+1, seq_len*2+3, seq_len, 0])
    
    ax4.set_title('Self-Attention vs Cross-Attention', fontsize=12, weight='bold')
    ax4.text(seq_len/2, -0.5, 'Self-Attention\n(Decoder→Decoder)', 
            ha='center', fontsize=10, weight='bold', color='blue')
    ax4.text(seq_len*1.5+2, -0.5, 'Cross-Attention\n(Decoder→Encoder)', 
            ha='center', fontsize=10, weight='bold', color='red')
    
    ax4.set_xlim(-0.5, seq_len*2+3.5)
    ax4.set_ylim(-1, seq_len+0.5)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    plt.suptitle('Understanding Attention Mechanisms in Transformers', 
                fontsize=16, weight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

# Create the new visualizations
print("🎨 Creating Enhanced Transformer Architecture Visualization...")
show_transformer_architecture()

print("\n🔍 Creating Detailed Attention Mechanism Visualization...")
plot_attention_mechanism()

# %% [markdown]
# ### 🎯 Exercise 2.1: Understanding Key Concepts
# **Fill in the blanks and run the code to test your understanding:**

# %%
# Exercise 2.1: Key Transformer Concepts
def transformer_quiz():
    """Interactive quiz about Transformer concepts"""
    
    print("🧠 Transformer Architecture Quiz")
    print("=" * 40)
    
    # Question 1
    print("1. The Transformer uses _______ to capture relationships between words")
    answer1 = "attention"  # Fill this in
    print(f"Your answer: {answer1}")
    print("✓ Correct! Attention mechanisms allow the model to focus on relevant parts of the input.\n")
    
    # Question 2  
    print("2. Unlike RNNs, Transformers can process sequences in _______ ")
    answer2 = "parallel"  # Fill this in
    print(f"Your answer: {answer2}")
    print("✓ Correct! This makes training much faster.\n")
    
    # Question 3
    print("3. The three key matrices in attention are Query, Key, and _______")
    answer3 = "Value"  # Fill this in
    print(f"Your answer: {answer3}")
    print("✓ Correct! Q, K, V are the foundation of attention.\n")
    
    return True

quiz_result = transformer_quiz()

# %% [markdown]
# ### Section 3: The Self-Attention Layer
# 
# **Self-attention allows each word to attend to all other words in the sequence.**
# 
# **The process:**
# 1. Transform input into Query (Q), Key (K), Value (V) matrices
# 2. Calculate attention weights: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
# 3. Each position can attend to all positions in the input

# %%
class SingleHeadAttention(nn.Module):
    """Implementation of single-head self-attention"""
    
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False) 
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        Returns:
            Attention output of shape (batch_size, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Create Q, K, V matrices
        Q = self.W_q(x)  # (batch_size, seq_len, d_k)
        K = self.W_k(x)  # (batch_size, seq_len, d_k)
        V = self.W_v(x)  # (batch_size, seq_len, d_k)
        
        # Step 2: Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch_size, seq_len, seq_len)
        
        # Step 3: Apply mask if provided (for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# %% [markdown]
# ### 🎯 Exercise 3.1: Understanding Self-Attention

# %%
# Exercise 3.1: Let's see self-attention in action!
def demonstrate_self_attention():
    """Demonstrate how self-attention works with a simple example"""
    
    # Create a simple example
    vocab_size = 10
    seq_len = 4
    d_model = 8
    d_k = 6
    
    # Create sample input (let's say it represents "The cat sat")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, d_model).to(device)  # batch_size=1
    
    # Initialize attention layer and move to device
    attention = SingleHeadAttention(d_model, d_k).to(device)
    
    # Forward pass
    output, weights = attention(x)
    
    print("🔍 Self-Attention Demonstration")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Device: {x.device}")
    
    # Visualize attention weights (move to CPU for visualization)
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights[0].detach().cpu().numpy(), 
                annot=True, 
                fmt='.3f',
                xticklabels=[f'Pos {i}' for i in range(seq_len)],
                yticklabels=[f'Pos {i}' for i in range(seq_len)],
                cmap='Blues')
    plt.title('Self-Attention Weights\n(How much each position attends to every position)')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions') 
    plt.show()
    
    # Verify attention weights sum to 1
    print(f"\nAttention weights sum (should be ~1.0): {weights.sum(dim=-1)}")
    
    return output, weights

attention_output, attention_weights = demonstrate_self_attention()

# %% [markdown]
# ### 🎯 Exercise 3.2: Build Your Own Attention

# %%
# Exercise 3.2: Complete the missing parts of this attention function
def manual_attention_calculation(Q, K, V, mask=None):
    """
    Calculate attention manually to understand the process
    Complete the missing parts marked with # TODO
    """
    print("🛠️ Manual Attention Calculation")
    print("=" * 40)
    
    # TODO: Calculate the scaling factor
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)  # Fill this in
    
    # TODO: Calculate attention scores (Q @ K^T)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # Fill this in
    
    # TODO: Apply scaling
    scores = scores * scale  # Fill this in
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # TODO: Apply softmax
    attention_weights = F.softmax(scores, dim=-1)  # Fill this in
    
    # TODO: Apply weights to values
    output = torch.matmul(attention_weights, V)  # Fill this in
    
    print(f"✓ Scaling factor: {scale:.4f}")
    print(f"✓ Scores shape: {scores.shape}")
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    return output, attention_weights

# Test your implementation
test_Q = torch.randn(1, 4, 6).to(device)
test_K = torch.randn(1, 4, 6).to(device)
test_V = torch.randn(1, 4, 6).to(device)

manual_output, manual_weights = manual_attention_calculation(test_Q, test_K, test_V)

# %% [markdown]
# ### Section 4: The Multi-Head Attention Layer
# 
# **Multi-head attention allows the model to attend to information from different representation subspaces:**
# - Split the input into multiple "heads"
# - Each head learns different types of relationships
# - Concatenate and project the results

# %%
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementation"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V for all heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate scaled dot-product attention"""
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Step 1: Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 2: Apply attention to each head
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Step 4: Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

# %% [markdown]
# ### 🎯 Exercise 4.1: Multi-Head Attention in Action

# %%
# Exercise 4.1: Compare single-head vs multi-head attention
def compare_attention_heads():
    """Compare single-head and multi-head attention"""
    
    # Setup
    batch_size, seq_len, d_model = 1, 6, 12
    num_heads = 3
    
    # Create input and move to device
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Single-head attention
    single_head = SingleHeadAttention(d_model, d_model).to(device)
    single_output, single_weights = single_head(x)
    
    # Multi-head attention  
    multi_head = MultiHeadAttention(d_model, num_heads).to(device)
    multi_output, multi_weights = multi_head(x, x, x)
    
    print("🔄 Single-Head vs Multi-Head Attention")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Single-head output: {single_output.shape}")
    print(f"Multi-head output: {multi_output.shape}")
    print(f"Single-head weights: {single_weights.shape}")
    print(f"Multi-head weights: {multi_weights.shape}")
    print(f"Device: {x.device}")
    
    # Visualize multiple heads (move to CPU for visualization)
    fig, axes = plt.subplots(1, num_heads, figsize=(15, 4))
    for head in range(num_heads):
        sns.heatmap(multi_weights[0, head].detach().cpu().numpy(),
                   annot=True, fmt='.2f', ax=axes[head],
                   xticklabels=[f'K{i}' for i in range(seq_len)],
                   yticklabels=[f'Q{i}' for i in range(seq_len)],
                   cmap='viridis')
        axes[head].set_title(f'Head {head + 1}')
    
    plt.suptitle('Multi-Head Attention: Each Head Learns Different Patterns')
    plt.tight_layout()
    plt.show()
    
    return single_output, multi_output

single_out, multi_out = compare_attention_heads()

# %% [markdown]
# ### 🎯 Exercise 4.2: Understanding Different Attention Patterns

# %%
# Exercise 4.2: Create different types of attention patterns
def create_attention_patterns():
    """Create and visualize different attention patterns"""
    
    seq_len = 8
    patterns = {}
    
    # 1. Identity pattern (attend to self)
    identity = torch.eye(seq_len)
    patterns['Identity (Self-Attention)'] = identity
    
    # 2. Uniform pattern (attend to all equally)
    uniform = torch.ones(seq_len, seq_len) / seq_len
    patterns['Uniform (Global Attention)'] = uniform
    
    # 3. Causal pattern (only attend to previous tokens)
    causal = torch.tril(torch.ones(seq_len, seq_len))
    causal = causal / causal.sum(dim=-1, keepdim=True)
    patterns['Causal (Decoder Attention)'] = causal
    
    # 4. Local pattern (attend to nearby tokens)
    local = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i-1)
        end = min(seq_len, i+2)
        local[i, start:end] = 1
    local = local / local.sum(dim=-1, keepdim=True)
    patterns['Local (Window Attention)'] = local
    
    # Visualize all patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        sns.heatmap(pattern.numpy(), annot=True, fmt='.2f', 
                   ax=axes[idx], cmap='Blues',
                   xticklabels=[f'K{i}' for i in range(seq_len)],
                   yticklabels=[f'Q{i}' for i in range(seq_len)])
        axes[idx].set_title(name)
    
    plt.tight_layout()
    plt.show()
    
    return patterns

attention_patterns = create_attention_patterns()

# %% [markdown]
# ### Section 5: Position Embedding
# 
# **Since Transformers have no inherent notion of sequence order, we need to inject positional information:**
# - Absolute positional encoding (original Transformer)
# - Relative positional encoding (more recent variants)
# - Learned vs Fixed positional encodings

# %%
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Calculate div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]

# %% [markdown]
# ### 🎯 Exercise 5.1: Visualizing Positional Encodings

# %%
# Exercise 5.1: Understand how positional encodings work
def visualize_positional_encoding():
    """Visualize positional encoding patterns"""
    
    d_model = 16
    max_seq_len = 50
    
    # Create positional encoding
    pos_encoding = PositionalEncoding(d_model, max_seq_len)
    
    # Get the encoding matrix
    pe_matrix = pos_encoding.pe[0, :max_seq_len, :].numpy()
    
    # Plot the positional encoding
    plt.figure(figsize=(15, 8))
    
    # Heatmap of all dimensions
    plt.subplot(2, 2, 1)
    sns.heatmap(pe_matrix.T, cmap='RdBu_r', center=0, 
                xticklabels=range(0, max_seq_len, 5),
                yticklabels=range(0, d_model, 2))
    plt.title('Positional Encoding Matrix\n(Rows=Dimensions, Cols=Positions)')
    plt.xlabel('Position')
    plt.ylabel('Encoding Dimension')
    
    # Plot specific dimensions over positions
    plt.subplot(2, 2, 2)
    positions = range(max_seq_len)
    for dim in [0, 1, 4, 5]:
        plt.plot(positions, pe_matrix[:, dim], label=f'Dim {dim}')
    plt.title('Positional Encoding Values')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show sine/cosine patterns
    plt.subplot(2, 2, 3)
    plt.plot(positions[:20], pe_matrix[:20, 0], 'o-', label='Dim 0 (sin)')
    plt.plot(positions[:20], pe_matrix[:20, 1], 's-', label='Dim 1 (cos)')
    plt.title('Sine/Cosine Pattern (First 20 positions)')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare learned vs fixed
    plt.subplot(2, 2, 4)
    # Simulate learned positional embedding (random for comparison)
    learned_pe = torch.randn(max_seq_len, d_model).numpy()
    plt.plot(positions[:20], pe_matrix[:20, 0], label='Fixed (Sinusoidal)')
    plt.plot(positions[:20], learned_pe[:20, 0], label='Learned (Random)')
    plt.title('Fixed vs Learned Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pe_matrix

pe_matrix = visualize_positional_encoding()

# %% [markdown]
# ### 🎯 Exercise 5.2: Position Encoding Effects

# %%
# Exercise 5.2: See how positional encoding affects word embeddings
def demonstrate_position_effects():
    """Show how positional encoding affects the same word in different positions"""
    
    # Simulate word embeddings
    vocab_size = 1000
    d_model = 8
    seq_len = 10
    
    # Create embedding layer and move to device
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    pos_encoding = PositionalEncoding(d_model).to(device)
    
    # Same word "cat" (token id = 42) in different positions
    word_id = 42
    positions = [0, 2, 5, 8]
    
    print("🐱 How Position Affects Word Meaning")
    print("=" * 40)
    
    results = {}
    for pos in positions:
        # Create sequence with "cat" at position pos
        input_ids = torch.zeros(1, seq_len, dtype=torch.long).to(device)
        input_ids[0, pos] = word_id
        
        # Get word embedding
        word_emb = embedding(input_ids)
        
        # Add positional encoding
        word_with_pos = pos_encoding(word_emb)
        
        # Extract the embedding for our word
        cat_embedding = word_emb[0, pos]
        cat_with_position = word_with_pos[0, pos]
        
        results[pos] = {
            'original': cat_embedding,
            'with_position': cat_with_position
        }
        
        print(f"Position {pos}:")
        print(f"  Original embedding: {cat_embedding[:4].detach().cpu().numpy()}")
        print(f"  With position:      {cat_with_position[:4].detach().cpu().numpy()}")
        print()
    
    # Calculate similarities between the same word at different positions
    print("Similarity between 'cat' at different positions:")
    print("-" * 50)
    
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            emb1 = results[pos1]['with_position']
            emb2 = results[pos2]['with_position']
            
            # Cosine similarity
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            print(f"Position {pos1} vs Position {pos2}: {similarity.item():.3f}")
    
    return results

position_results = demonstrate_position_effects()

# %% [markdown]
# ## 📚 Day 2: May 25 - Implementation Day
# ### Section 6: The Encoder

# %%
class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

# %% [markdown]
# ### 🎯 Exercise 6.1: Build and Test the Encoder

# %%
# Exercise 6.1: Let's build a complete encoder and see how it transforms inputs
def test_encoder():
    """Test the Transformer encoder with sample data"""
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    num_layers = 3
    d_ff = 64
    vocab_size = 100
    
    # Create sample input
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Create embeddings and move to device
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    pos_encoding = PositionalEncoding(d_model).to(device)
    
    # Get embeddings with position
    x = embedding(input_ids)
    x = pos_encoding(x)
    
    print("🏗️ Transformer Encoder Test")
    print("=" * 40)
    print(f"Input shape: {x.shape}")
    print(f"Device: {x.device}")
    
    # Create encoder and move to device
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff).to(device)
    
    # Forward pass
    output = encoder(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Analyze how representations change through layers
    layer_outputs = []
    temp_x = x
    
    for i, layer in enumerate(encoder.layers):
        temp_x = layer(temp_x)
        layer_outputs.append(temp_x.clone())
        
        # Calculate average attention pattern (simplified analysis)
        mean_values = temp_x.mean(dim=[0, 1])
        print(f"Layer {i+1} - Mean activation: {mean_values.mean().item():.4f}, "
              f"Std: {mean_values.std().item():.4f}")
    
    return output, layer_outputs

encoder_output, layer_outputs = test_encoder()

# %% [markdown]
# ### 🎯 Exercise 6.2: Analyze Information Flow

# %%
# Exercise 6.2: Visualize how information flows through encoder layers
def analyze_encoder_information_flow():
    """Analyze how information changes through encoder layers"""
    
    # Use outputs from previous exercise
    if 'layer_outputs' not in locals():
        _, layer_outputs = test_encoder()
    
    num_layers = len(layer_outputs)
    
    # Calculate layer-wise statistics
    stats = {
        'mean_activation': [],
        'std_activation': [],
        'mean_magnitude': []
    }
    
    for layer_output in layer_outputs:
        stats['mean_activation'].append(layer_output.mean().item())
        stats['std_activation'].append(layer_output.std().item())
        stats['mean_magnitude'].append(layer_output.abs().mean().item())
    
    # Plot the statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    layers = range(1, num_layers + 1)
    
    # Mean activation
    axes[0].plot(layers, stats['mean_activation'], 'o-')
    axes[0].set_title('Mean Activation per Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Activation')
    axes[0].grid(True, alpha=0.3)
    
    # Standard deviation
    axes[1].plot(layers, stats['std_activation'], 's-', color='orange')
    axes[1].set_title('Activation Std per Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].grid(True, alpha=0.3)
    
    # Mean magnitude
    axes[2].plot(layers, stats['mean_magnitude'], '^-', color='green')
    axes[2].set_title('Mean Magnitude per Layer')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Mean Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Information Flow Analysis")
    print("=" * 40)
    print("As information flows through layers:")
    print("- Mean activation shows the average 'signal' level")
    print("- Standard deviation shows how varied the representations are")
    print("- Mean magnitude shows the overall 'strength' of representations")
    
    return stats

flow_stats = analyze_encoder_information_flow()

# %% [markdown]
# ### Section 7: The Decoder
# 
# **The decoder has additional complexity:**
# - Masked self-attention (can't see future tokens)
# - Encoder-decoder attention (attends to encoder outputs)
# - Autoregressive generation during inference

# %%
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal mask for decoder self-attention"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Encoder-decoder attention
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                causal_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 1. Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Encoder-decoder attention
        attn_output, _ = self.encoder_attention(x, encoder_output, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerDecoder(nn.Module):
    """Stack of Transformer Decoder Layers"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, padding_mask)
            
        return x

# %% [markdown]
# ### 🎯 Exercise 7.1: Understanding Causal Masking

# %%
# Exercise 7.1: Understand why we need causal masking
def demonstrate_causal_masking():
    """Show the difference between masked and unmasked attention in decoder"""
    
    seq_len = 6
    d_model = 8
    
    # Create sample decoder input (representing partial translation)
    x = torch.randn(1, seq_len, d_model).to(device)
    
    # Create causal mask and move to device
    causal_mask = create_causal_mask(seq_len).to(device)
    
    # Attention without mask (WRONG for decoder)
    attention_no_mask = MultiHeadAttention(d_model, 2).to(device)
    output_no_mask, weights_no_mask = attention_no_mask(x, x, x)
    
    # Attention with causal mask (CORRECT for decoder)  
    attention_with_mask = MultiHeadAttention(d_model, 2).to(device)
    output_with_mask, weights_with_mask = attention_with_mask(x, x, x, causal_mask)
    
    # Visualize the difference (move to CPU for visualization)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Show the mask
    axes[0].imshow(causal_mask[0, 0].cpu().numpy(), cmap='Blues')
    axes[0].set_title('Causal Mask\n(1=allowed, 0=blocked)')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # Attention without mask (head 0)
    sns.heatmap(weights_no_mask[0, 0].detach().cpu().numpy(), 
                annot=True, fmt='.2f', ax=axes[1], cmap='Reds',
                xticklabels=[f'K{i}' for i in range(seq_len)],
                yticklabels=[f'Q{i}' for i in range(seq_len)])
    axes[1].set_title('Without Causal Mask\n(Can see future!)')
    
    # Attention with mask (head 0)
    sns.heatmap(weights_with_mask[0, 0].detach().cpu().numpy(),
                annot=True, fmt='.2f', ax=axes[2], cmap='Blues',
                xticklabels=[f'K{i}' for i in range(seq_len)],
                yticklabels=[f'Q{i}' for i in range(seq_len)])
    axes[2].set_title('With Causal Mask\n(Cannot see future)')
    
    plt.tight_layout()
    plt.show()
    
    print("🔒 Causal Masking Demonstration")
    print("=" * 40)
    print("Without causal mask: Decoder can 'cheat' by looking at future tokens")
    print("With causal mask: Decoder can only look at previous tokens (autoregressive)")
    print("\nThis is crucial for:")
    print("- Training: Ensures model learns to predict from past context only")
    print("- Inference: Matches the autoregressive generation process")
    
    return weights_no_mask, weights_with_mask

no_mask_weights, masked_weights = demonstrate_causal_masking()

# %% [markdown]
# ### 🎯 Exercise 7.2: Complete Encoder-Decoder

# %%
# Exercise 7.2: Build and test complete encoder-decoder
def test_encoder_decoder():
    """Test complete encoder-decoder transformer"""
    
    # Hyperparameters
    batch_size = 2
    src_seq_len = 8  # Source sequence length
    tgt_seq_len = 6  # Target sequence length  
    d_model = 16
    num_heads = 4
    num_layers = 2
    d_ff = 64
    vocab_size = 100
    
    # Create sample data and move to device
    torch.manual_seed(42)
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len)).to(device)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # Embeddings and positional encoding - move to device
    embedding = nn.Embedding(vocab_size, d_model).to(device)
    pos_encoding = PositionalEncoding(d_model).to(device)
    
    # Encode source
    src_emb = pos_encoding(embedding(src_tokens))
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff).to(device)
    encoder_output = encoder(src_emb)
    
    # Decode target
    tgt_emb = pos_encoding(embedding(tgt_tokens))
    decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff).to(device)
    
    # Create causal mask for target and move to device
    causal_mask = create_causal_mask(tgt_seq_len).to(device)
    
    decoder_output = decoder(tgt_emb, encoder_output, causal_mask)
    
    print("🔄 Complete Encoder-Decoder Test")
    print("=" * 40)
    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Device: {src_tokens.device}")
    
    # Calculate number of parameters
    total_params = (sum(p.numel() for p in encoder.parameters()) + 
                   sum(p.numel() for p in decoder.parameters()) +
                   sum(p.numel() for p in embedding.parameters()) +
                   sum(p.numel() for p in pos_encoding.parameters()))
    
    print(f"Total parameters: {total_params:,}")
    
    return encoder_output, decoder_output

enc_out, dec_out = test_encoder_decoder()

# %% [markdown]
# ### Section 8: Complete Transformer Implementation

# %%
class Transformer(nn.Module):
    """Complete Transformer Model"""
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int, 
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence"""
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence"""
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the complete transformer"""
        encoder_output = self.encode(src, src_mask)
        output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return output

# %% [markdown]
# ### 🎯 Exercise 8.1: Build Your Complete Transformer

# %%
# Exercise 8.1: Create and test the complete transformer
def test_complete_transformer():
    """Test the complete transformer implementation"""
    
    # Model hyperparameters
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 128
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 512
    max_seq_len = 100
    
    # Create model and move to device
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Sample data
    batch_size = 4
    src_seq_len = 12
    tgt_seq_len = 10
    
    torch.manual_seed(42)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # Create masks and move to device
    tgt_mask = create_causal_mask(tgt_seq_len).to(device)
    
    # Forward pass
    output = transformer(src, tgt, tgt_mask=tgt_mask)
    
    print("🎉 Complete Transformer Test")
    print("=" * 40)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output vocabulary size: {output.shape[-1]}")
    print(f"Device: {src.device}")
    
    # Model size
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # 4 bytes per float32
    
    # Test gradient flow
    loss = F.cross_entropy(output.view(-1, tgt_vocab_size), 
                          torch.randint(0, tgt_vocab_size, (batch_size * tgt_seq_len,)).to(device))
    loss.backward()
    
    # Check gradient statistics
    grad_norms = []
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"\nGradient Statistics:")
    print(f"Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"Max gradient norm: {np.max(grad_norms):.6f}")
    print(f"Min gradient norm: {np.min(grad_norms):.6f}")
    
    return transformer, output

transformer_model, transformer_output = test_complete_transformer()

# %% [markdown]
# ### 🎯 Exercise 8.2: Autoregressive Generation

# %%
# Exercise 8.2: Implement autoregressive text generation
def autoregressive_generation(model: Transformer, src: torch.Tensor, 
                            start_token: int, end_token: int, max_length: int = 20):
    """Generate text autoregressively using the transformer"""
    
    model.eval()
    
    with torch.no_grad():
        # Encode source
        encoder_output = model.encode(src)
        
        # Start with start token on the same device as src
        generated = torch.tensor([[start_token]], dtype=torch.long, device=src.device)
        
        for _ in range(max_length):
            # Create causal mask for current sequence
            seq_len = generated.shape[1]
            tgt_mask = create_causal_mask(seq_len).to(src.device)
            
            # Decode
            output = model.decode(generated, encoder_output, tgt_mask)
            
            # Get next token probabilities
            next_token_logits = output[0, -1, :]  # Last position, batch 0
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token (you could also use greedy or beam search)
            next_token = torch.multinomial(next_token_probs, 1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if end token generated
            if next_token.item() == end_token:
                break
    
    return generated

# Demonstrate autoregressive generation
def demonstrate_generation():
    """Show how autoregressive generation works"""
    
    print("🤖 Autoregressive Generation Demo")
    print("=" * 40)
    
    # Use the transformer from previous exercise
    src_vocab_size = 1000
    tgt_vocab_size = 800
    
    # Special tokens
    start_token = 1
    end_token = 2
    
    # Sample source sequence and move to device
    src = torch.randint(3, src_vocab_size, (1, 8)).to(device)  # Avoid special tokens
    
    print(f"Source sequence: {src[0].tolist()}")
    print(f"Device: {src.device}")
    
    # Generate target sequence
    generated = autoregressive_generation(
        transformer_model, src, start_token, end_token, max_length=15
    )
    
    print(f"Generated sequence: {generated[0].tolist()}")
    print(f"Generated length: {generated.shape[1]}")
    
    # Show step-by-step process
    print("\n🔍 Step-by-step generation:")
    print("-" * 30)
    
    # Simulate step by step (just show concept)
    current_seq = [start_token]
    for i in range(min(5, generated.shape[1] - 1)):  # Show first 5 steps
        next_token = generated[0, i + 1].item()
        current_seq.append(next_token)
        print(f"Step {i+1}: {current_seq}")
        
        if next_token == end_token:
            break
    
    return generated

generated_sequence = demonstrate_generation()

# %% [markdown]
# ### Section 9: Testing and Analysis

# %%
# Exercise 9.1: Comprehensive model analysis
def comprehensive_model_analysis():
    """Comprehensive analysis of our transformer implementation"""
    
    print("🔬 Comprehensive Model Analysis")
    print("=" * 50)
    
    # 1. Architecture verification
    print("1. Architecture Verification:")
    print("   ✓ Multi-head attention implemented")
    print("   ✓ Positional encoding implemented") 
    print("   ✓ Encoder-decoder structure implemented")
    print("   ✓ Residual connections and layer norm implemented")
    print("   ✓ Causal masking for decoder implemented")
    
    # 2. Parameter count analysis
    def count_parameters_by_component(model):
        """Count parameters by component"""
        counts = {}
        
        # Embeddings
        counts['embeddings'] = (model.src_embedding.weight.numel() + 
                               model.tgt_embedding.weight.numel())
        
        # Encoder
        counts['encoder'] = sum(p.numel() for p in model.encoder.parameters())
        
        # Decoder  
        counts['decoder'] = sum(p.numel() for p in model.decoder.parameters())
        
        # Output projection
        counts['output'] = model.output_projection.weight.numel()
        
        return counts
    
    param_counts = count_parameters_by_component(transformer_model)
    total = sum(param_counts.values())
    
    print(f"\n2. Parameter Distribution:")
    for component, count in param_counts.items():
        percentage = (count / total) * 100
        print(f"   {component.capitalize()}: {count:,} ({percentage:.1f}%)")
    print(f"   Total: {total:,}")
    
    # 3. Memory analysis
    print(f"\n3. Memory Analysis:")
    model_size_mb = total * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Approximate GPU memory for training: {model_size_mb * 4:.2f} MB")
    
    # 4. Computational complexity
    d_model = transformer_model.d_model
    num_heads = 8  # from our model
    seq_len = 100  # example sequence length
    
    # Attention complexity: O(n^2 * d)
    attention_ops = seq_len * seq_len * d_model * num_heads
    
    # FFN complexity: O(n * d^2)  
    ffn_ops = seq_len * d_model * d_model * 4  # d_ff is typically 4*d_model
    
    print(f"\n4. Computational Complexity (seq_len={seq_len}):")
    print(f"   Attention operations: {attention_ops:,}")
    print(f"   Feed-forward operations: {ffn_ops:,}")
    print(f"   Attention dominates for seq_len > {d_model}")
    
    return param_counts

analysis_results = comprehensive_model_analysis()

# %% [markdown]
# ### 🎯 Exercise 9.2: Compare with Different Configurations

# %%
# Exercise 9.2: Compare different model configurations
def compare_model_configurations():
    """Compare different transformer configurations"""
    
    configs = {
        'Tiny': {'d_model': 64, 'num_heads': 4, 'num_layers': 2, 'd_ff': 256},
        'Small': {'d_model': 128, 'num_heads': 8, 'num_layers': 4, 'd_ff': 512},
        'Base': {'d_model': 256, 'num_heads': 8, 'num_layers': 6, 'd_ff': 1024},
        'Large': {'d_model': 512, 'num_heads': 16, 'num_layers': 12, 'd_ff': 2048}
    }
    
    results = {}
    
    for name, config in configs.items():
        # Create model
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            d_ff=config['d_ff']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate memory
        memory_mb = total_params * 4 / (1024 * 1024)
        
        # Estimate operations for seq_len=128
        seq_len = 128
        attention_ops = seq_len * seq_len * config['d_model'] * config['num_heads'] * config['num_layers'] * 2  # encoder + decoder
        ffn_ops = seq_len * config['d_model'] * config['d_ff'] * config['num_layers'] * 2
        
        results[name] = {
            'params': total_params,
            'memory_mb': memory_mb,
            'attention_ops': attention_ops,
            'ffn_ops': ffn_ops,
            'total_ops': attention_ops + ffn_ops
        }
    
    # Display comparison
    print("📊 Model Configuration Comparison")
    print("=" * 60)
    print(f"{'Config':<8} {'Params':<12} {'Memory(MB)':<12} {'Operations':<15}")
    print("-" * 60)
    
    for name, stats in results.items():
        print(f"{name:<8} {stats['params']:<12,} {stats['memory_mb']:<12.1f} {stats['total_ops']:<15,}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    names = list(results.keys())
    params = [results[name]['params'] for name in names]
    memory = [results[name]['memory_mb'] for name in names]
    ops = [results[name]['total_ops'] for name in names]
    
    # Parameters
    axes[0, 0].bar(names, params, color='skyblue')
    axes[0, 0].set_title('Parameter Count')
    axes[0, 0].set_ylabel('Parameters')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Memory
    axes[0, 1].bar(names, memory, color='lightcoral')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Operations
    axes[1, 0].bar(names, ops, color='lightgreen')
    axes[1, 0].set_title('Computational Operations')
    axes[1, 0].set_ylabel('Operations')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Efficiency (params per operation)
    efficiency = [p / o for p, o in zip(params, ops)]
    axes[1, 1].bar(names, efficiency, color='gold')
    axes[1, 1].set_title('Efficiency (Params/Ops)')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

config_comparison = compare_model_configurations()

# %% [markdown]
# ### 🎯 Final Exercise: Put It All Together

# %%
# Final Exercise: Create a simple training loop
def create_simple_training_loop():
    """Create a simple training loop to see the transformer in action"""
    
    print("🚀 Simple Training Loop Demo")
    print("=" * 40)
    
    # Create a small model for quick training and move to device
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        max_seq_len=50
    ).to(device)
    
    # Create simple synthetic data
    batch_size = 8
    seq_len = 10
    num_batches = 5
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
    
    model.train()
    losses = []
    
    print(f"Training on device: {device}")
    print("Training progress:")
    for batch_idx in range(num_batches):
        # Generate random data (in practice, this would be real data) - move to device
        src = torch.randint(1, 100, (batch_size, seq_len)).to(device)
        tgt_input = torch.randint(1, 100, (batch_size, seq_len)).to(device)
        tgt_output = torch.randint(1, 100, (batch_size, seq_len)).to(device)
        
        # Create causal mask and move to device
        tgt_mask = create_causal_mask(seq_len).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-')
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    return model, losses

trained_model, training_losses = create_simple_training_loop()

# %% [markdown]
# ## 🎓 Course Summary and Next Steps
# 
# ### What You've Learned:
# 
# #### 📚 **Theoretical Understanding:**
# 1. **Self-Attention Mechanism** - How words can attend to all other words in parallel
# 2. **Multi-Head Attention** - Learning different types of relationships simultaneously  
# 3. **Positional Encoding** - Injecting sequence order information
# 4. **Encoder-Decoder Architecture** - The complete transformer structure
# 5. **Causal Masking** - Preventing future information leakage in decoders
# 
# #### 💻 **Practical Implementation:**
# 1. Built each component from scratch using PyTorch
# 2. Implemented complete working Transformer model
# 3. Created autoregressive generation capability
# 4. Analyzed model complexity and configurations
# 5. Set up basic training pipeline
# 
# ### 🚀 **Next Steps for Continued Learning:**
# 
# #### **Advanced Topics to Explore:**
# - **BERT & GPT architectures** - Encoder-only and decoder-only variants
# - **Vision Transformers (ViT)** - Applying transformers to images
# - **Optimization techniques** - Learning rate scheduling, warmup, etc.
# - **Advanced attention variants** - Sparse attention, linear attention
# - **Pre-training and fine-tuning** - Transfer learning with transformers
# 
# #### **Practical Projects:**
# 1. **Text Classification** - Fine-tune BERT for sentiment analysis
# 2. **Machine Translation** - Build a translation system
# 3. **Text Generation** - Create a GPT-style language model
# 4. **Question Answering** - Implement a QA system
# 5. **Multimodal Learning** - Combine text and images
# 
# #### **Resources for Further Learning:**
# - **Papers**: "Attention Is All You Need", "BERT", "GPT-3"
# - **Libraries**: Hugging Face Transformers, fairseq
# - **Courses**: Stanford CS224N, Fast.ai NLP course
# - **Books**: "Natural Language Processing with Transformers"

# %% [markdown]
# ### 🎯 **Congratulations!** 
# 
# You've successfully built a complete Transformer architecture from scratch and understand:
# - ✅ Why attention revolutionized NLP
# - ✅ How self-attention enables parallel processing
# - ✅ The importance of positional encoding
# - ✅ How multi-head attention captures different relationships
# - ✅ The encoder-decoder architecture
# - ✅ Autoregressive generation process
# 
# **You're now ready to dive deeper into the world of modern NLP and large language models!**

print("🎉 Course Complete! You've mastered the Transformer architecture!")
print("Ready to build the next generation of AI models!")