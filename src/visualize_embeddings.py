import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

tsne_dimension_reducer = TSNE(n_components=2)
pca_dimension_reducer = PCA(n_components=2)


def visualize_augmentations(anchor_df, positive_df, negative_df, output_dir,
                            title):
  _, axs = plt.subplots(
      ncols=2, nrows=2, figsize=(12, 12), constrained_layout=True)
  # add an artist, in this case a nice label in the middle...
  augmentation_dfs = [positive_df, negative_df]
  augmentation_names = ['Positive', 'Negative']
  for row in range(2):
    label_to_plot = row
    anchor_label_df = anchor_df[anchor_df.label == label_to_plot]
    anchor_label_df['augmentation'] = 'Original'
    for col in range(2):
      augmentation_df = augmentation_dfs[col]
      augmentation_df = augmentation_df[augmentation_df.label == label_to_plot]
      augmentation_df['augmentation'] = augmentation_names[col]
      plot_df = pd.concat([augmentation_df, anchor_label_df], ignore_index=True)

      ax = axs[row, col]
      sns.scatterplot(data=plot_df, x='x', y='y', hue='augmentation', ax=ax)
  plt.savefig(
      f'{output_dir}/{title}_augmentations.png', format='png', pad_inches=0)


def reduce_embedding_dimensions(embeddings,
                                masks,
                                labels,
                                dimension_reducer=tsne_dimension_reducer):
  labels = labels.numpy().reshape(-1)

  averaged_hidden_states = torch.div(
      embeddings.sum(dim=1), masks.sum(dim=1, keepdim=True))
  reduced_dimension_embeddings = dimension_reducer.fit_transform(
      averaged_hidden_states.numpy())

  df = pd.DataFrame.from_dict({
      'x': reduced_dimension_embeddings[:, 0],
      'y': reduced_dimension_embeddings[:, 1],
      'label': labels
  })
  return df


def visualize(df_to_visualize, output_dir, title):
  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_axes()
  sns.scatterplot(data=df_to_visualize, x='x', y='y', hue='label', ax=ax)
  plt.savefig(f'{output_dir}/{title}.png', format='png', pad_inches=0)
