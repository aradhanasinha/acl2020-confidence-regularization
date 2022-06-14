import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
  """
    Calculates the contrastive loss loss for self-supervised learning.
    Elsewhere referred to as InfoNCE loss or NTXentLoss.

    Returns:
         Value of the Contrastive Loss.
     Examples:
        >>> loss = ContrastiveLoss()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> anchor = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = [torch.randn(batch_size, embedding_size) * 3]
        >>> output = loss(anchor, positive_key, negative_keys)
  """

  def __init__(self,
               temperature=0.1,
               use_unpaired_negative_keys=True, 
               reduction='mean',):
    super().__init__()
    self.temperature = temperature
    self.reduction = reduction
    self.use_unpaired_negative_keys = use_unpaired_negative_keys

  def forward(self, anchor_embedding, shuffled_embeddings_list,
              token_drop_embeddings_list):
    """
    Args:
      anchor_embedding: batch embeddings (len N)
      shuffled_embeddings_list: each item in the list are shuffled embeddings
        for the entire batch (len N). The length of the shuffled embeddings is
        variable, and currently set by create_input_features fn. This is the 
        negative augmentation.
      token_drop_embeddings_list: same as above but for token drop. Intended
        as the positve augmentation.
      Each batch-embedding has size (batch_size, sequence_length, hidden_size)

    Returns: the loss
    """

    # Let D be sequence_length * hidden_size. This rehapes into (N, D)
    reshape_embedding = lambda e: e.view(e.shape[0], -1)

    anchor_embedding = reshape_embedding(anchor_embedding)
    positive_embedding = reshape_embedding(token_drop_embeddings_list[0])

    negative_embeddings = [
        reshape_embedding(x) for x in shuffled_embeddings_list
    ]
    negative_embedding_tensor = torch.Tensor(
        len(negative_embeddings), *positive_embedding.shape)
    torch.cat(negative_embeddings, out=negative_embedding_tensor)  # (M, N, D)
    negative_embedding_tensor = negative_embedding_tensor.transpose(
        0, 1)  # (N, M, D)

    return contrastive_loss(
        anchor_embedding,
        positive_embedding,
        negative_embedding_tensor,
        use_unpaired_negative_keys=self.use_unpaired_negative_keys,
        temperature=self.temperature,
        reduction=self.reduction)


def contrastive_loss(anchor,
                     positive_key,
                     negative_keys_paired=None,
                     use_unpaired_negative_keys=True,
                     temperature=0.1,
                     reduction='mean'):
  """
    Calculates the contrastive loss loss for self-supervised learning.
    Elsewhere referred to as InfoNCE loss or NTXentLoss.

    This contrastive loss enforces the embeddings of similar (positive) samples
    to be close
        and those of different (negative) samples to be distant.
    A anchor embedding is compared with one positive key and with one or more
    negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the
          cross entropy.
        reduction: Reduction method applied to the output. Value must be one of
          ['none', 'sum', 'mean']. See torch.nn.functional.cross_entropy for
          more details about each option.
    Input shape:
        anchor: (N, D) Tensor with anchor samples (e.g. embeddings of the
          input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of
          augmented input).
        negative_keys_paired (optional): Tensor with negative samples (e.g. 
          embeddings of other inputs) If negative_mode = 'paired', then 
          negative_keys is a (N, M, D) Tensor.
        use_unpaired_negative_keys: If true, adds negative keys. The additional
          negative keys for a sample are the positive keys for the other
          samples.

  Returns: contrastive loss value

  Raises:
    ValueError: if arguments violate size assumptions or if no negative 
        examples are speficied (use_unpaired_negative_keys is False and 
        negative_keys_paired is None)
  """
  check_input_dimensions(
      anchor=anchor,
      positive_key=positive_key,
      negative_keys_paired=negative_keys_paired)
  if negative_keys_paired is None and not use_unpaired_negative_keys:
    raise ValueError('Need negative examples: paired, unpaired or both.')

  # Normalize D length embeddigngs to unit vectors, so we can dot product away.
  anchor, positive_key, negative_keys_paired = normalize_last_dim(
      anchor, positive_key, negative_keys_paired)

  if negative_keys_paired is not None:
    # Cosine between all anchor-negative combinations
    # (N, 1, M) <-- (N, 1, D) * (N, D, M)
    negative_logits = anchor.unsqueeze(1) @ transpose_last_two_dim(
        negative_keys_paired)
    negative_logits = negative_logits.squeeze(1)  # (N, M)

  if not use_unpaired_negative_keys:
    # Cosine between positive pairs
    # (N, 1) <-- (N, D) dot.prod (N, D), but keep dimension
    positive_logit = torch.sum(anchor * positive_key, dim=1, keepdim=True)

    # First index in last dimension are the positive samples
    # (N, M+1) <-- [positive, negative]
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    # Correct label is at the 0th index always
    labels = torch.zeros(len(logits), dtype=torch.long, device=anchor.device)
  else:
    # Negative keys are implicitly off-diagonal positive keys.

    # Cosine between all combinations
    # (N, N) <-- (N, D) * (D, N) (positive on diagonal)
    logits = anchor @ transpose_last_two_dim(positive_key)

    if negative_keys_paired is not None:
      # (N, N + M)  <-- [logits, negative]
      logits = torch.cat([logits, negative_logits], dim=1)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(anchor), device=anchor.device)

  return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def check_input_dimensions(anchor,
                           positive_key,
                           negative_keys_paired=None):
  """Check input dimensionality."""
  if anchor.dim() != 2:
    raise ValueError('<anchor> must have 2 dimensions.')
  if positive_key.dim() != 2:
    raise ValueError('<positive_key> must have 2 dimensions.')
  if negative_keys_paired is not None:
    if negative_keys_paired.dim() != 3:
      raise ValueError('<negative_keys_paired> must have 3 dimensions if set.')

  # Check matching number of samples.
  if len(anchor) != len(positive_key):
    raise ValueError(
        '<anchor> and <positive_key> must must have the same number of samples.'
    )
  if negative_keys_paired is not None:
    if len(anchor) != len(negative_keys_paired):
      raise ValueError(
          '<negative_keys_paired> must have the same number of samples as <anchor> if set.'
      )

  # Embedding vectors should have same number of components.
  if anchor.shape[-1] != positive_key.shape[-1]:
    raise ValueError(
        'Vectors of <anchor> and <positive_key> should have the same number of components.'
    )
  if negative_keys_paired is not None:
    if anchor.shape[-1] != negative_keys_paired.shape[-1]:
      raise ValueError(
          'Vectors of <anchor> and <negative_keys_paired> should have the same number of components.'
      )
  return


def transpose_last_two_dim(x):
  return x.transpose(-2, -1)


def normalize_last_dim(*xs):
  return [None if x is None else F.normalize(x, dim=-1) for x in xs]
