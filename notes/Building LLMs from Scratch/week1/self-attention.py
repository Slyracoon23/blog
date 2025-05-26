# %% [raw]
# ---
# title: "EXERCISE: Self-Attention"
# categories: Building LLMs from Scratch
# date: 05-25-2025
# ---

# %% [markdown]
# # Self-Attention
# 
# Welcome to the first exercise in our Building LLMs from Scratch series! In this exercise, we'll dive into the core concept of self-attention, which is the foundation of modern transformer-based models.
# 
# ## 📚 Learning Path:
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # d_in: Input dimension - the size of each input embedding/feature vector
        self.d_in = d_in
        # d_out: Output dimension - the desired size of the attention output
        self.d_out = d_out

        # Linear transformation layers for generating queries, keys, and values
        # q: Query projection - transforms input to query space for "what am I looking for?"
        self.q = nn.Linear(d_in, d_out)
        # k: Key projection - transforms input to key space for "what information do I contain?"
        self.k = nn.Linear(d_in, d_out)
        # v: Value projection - transforms input to value space for "what information should I output?"
        self.v = nn.Linear(d_in, d_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Input tensor of shape (batch_size, sequence_length, d_in)
        # Generate query vectors - represent what each position is "looking for"
        q = self.q(x)
        # Generate key vectors - represent what information each position "contains"
        k = self.k(x)
        # Generate value vectors - represent the actual information to be aggregated
        v = self.v(x)

        # Calculate attention scores - how much each position is relevant to each other
        scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.d_out))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # Apply attention weights to value vectors to get aggregated output
        hidden_states = torch.bmm(attn_weights, v)
        return hidden_states



# %% [markdown]
# Now let's add the SOS token to the input and EOT token to the output.
# %%
SOS_TOKEN = "<SOS>"
EOT_TOKEN = "<EOT>"

words = ["hello", "world", "this", "is", "a", "test"]

# %% [markdown]
# Index mapping for the words
# %%
word_set_idx = {i: word for i, word in enumerate(words + [SOS_TOKEN, EOT_TOKEN])}

print(word_set_idx)

# %% [markdown]
# Now let's invert the map to have keys be the words and values be the indices.
# %%
word_to_ix = {word: i for i, word in enumerate(words + [SOS_TOKEN, EOT_TOKEN])}

print(word_to_ix)

# %% [markdown]
# Now let's create a helper function to convert a list of words to a tensor of input tokens.
# %%
def convert_words_to_tensors(words: list[str]) -> torch.Tensor:
    return torch.tensor([word_to_ix[word] for word in words], dtype=torch.long).view(-1, 1)

# %% [markdown]
# Now let's create a helper function to convert a tensor of input tokens to a list of words.
# %%
def convert_tensors_to_words(tensors: torch.Tensor) -> list[str]:
    return [word_set_idx[i] for i in tensors]

# %% [markdown]