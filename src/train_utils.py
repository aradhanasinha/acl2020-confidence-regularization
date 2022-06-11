"""
Utils for train*.py.
"""
import random
from typing import Dict, Iterable, List
from functools import total_ordering
from collections import namedtuple
from utils import Processor, process_par
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler

import nltk
import torch

nltk.download("punkt")

# FEVER: claim --> hypothesis,  evidence --> premise
TextPairExample = namedtuple("TextPairExample",
                             ["id", "premise", "hypothesis", "label"])

InputFeature = namedtuple("InputFeature",
                          ["input_ids", "segment_ids", "label_id"])

@total_ordering
class InputFeatures(object):
  """A single set of features of data with its augmentations."""
  ORIGINAL_INPUT = "original_input_feature"
  SHUFFLED_INPUT = "shuffled_input_feature"

  def __init__(self, example_id, original_input, shuffled_input, bias):
    self.example_id = example_id
    self.input_features_dict = {
        self.ORIGINAL_INPUT: original_input,
        self.SHUFFLED_INPUT: shuffled_input
    }
    self.bias = bias

  def _get_key(self):
    """Preserves the original code's key used for sorting this class."""
    return self.input_features_dict.get(self.ORIGINAL_INPUT).input_ids

  def __eq__(self, other):
    return self._get_key() == other._get_key()

  def __lt__(self, other):
    return self._get_key() < other._get_key()

  def get_original_label_id(self):
    return self.input_features_dict[self.ORIGINAL_INPUT].label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class ExampleConverter(Processor):

  def __init__(self, max_seq_length, tokenizer):
    self.max_seq_length = max_seq_length
    self.tokenizer = tokenizer

  def _get_input_and_segment_ids(self, premise, hypothesis=None):
    tokens_a = self.tokenizer.tokenize(premise)
    tokens_b = None
    if hypothesis:
      tokens_b = self.tokenizer.tokenize(hypothesis)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > self.max_seq_length - 2:
        tokens_a = tokens_a[:(self.max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    return input_ids, segment_ids


  @staticmethod
  def _shuffle_sentence_like_word_ordering_paper(sentence):
    if len(nltk.sent_tokenize(sentence)) > 1:
      return None
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) < 3:
      return None
    random.Random(10).shuffle(tokens)
    return " ".join(tokens)

  def create_input_features(self, text_pair_example):
    original_input_ids, original_segment_ids = self._get_input_and_segment_ids(
        text_pair_example.premise, text_pair_example.hypothesis)
    original_input_feature = InputFeature(
        input_ids=original_input_ids,
        segment_ids=original_segment_ids,
        label_id=text_pair_example.label)

    shuffled_hypothesis = ExampleConverter._shuffle_sentence_like_word_ordering_paper(
        text_pair_example.hypothesis)
    shuffled_input_ids, shuffled_segment_ids = self._get_input_and_segment_ids(
        text_pair_example.premise, shuffled_hypothesis)
    shuffled_input_feature = InputFeature(
        input_ids=shuffled_input_ids,
        segment_ids=shuffled_segment_ids,
        label_id=-1)

    return InputFeatures(
        example_id=text_pair_example.id,
        original_input=original_input_feature,
        shuffled_input=shuffled_input_feature,
        bias=None)

  def process(self, data: Iterable):
    features = []
    for example in data:
      features.append(self.create_input_features(example))
    return features


class InputFeatureDataset(Dataset):

  def __init__(self, examples: List[InputFeatures]):
    self.examples = examples

  def __getitem__(self, index):
    return self.examples[index]

  def __len__(self):
    return len(self.examples)


def collate_input_features(batch: List[InputFeatures]):
  """Collate InputFeatures for batch.
  Returns a list of tensors.
  """
  sz = len(batch)

  input_feature_names = set.intersection(
      *map(set, [x.input_features_dict for x in batch]))

  max_seq_len = 0
  for input_feature_name in input_feature_names:
    max_seq_len_feature = max(
        len(x.input_features_dict[input_feature_name].input_ids) for x in batch)
    max_seq_len = max(max_seq_len, max_seq_len_feature)

  input_features_collated_dict = {}
  for input_feature_name in input_feature_names:
    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
      input_feature = ex.input_features_dict[input_feature_name]
      input_ids[i, :len(input_feature.input_ids)] = input_feature.input_ids
      segment_ids[
          i, :len(input_feature.segment_ids)] = input_feature.segment_ids
      mask[i, :len(input_feature.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(
        np.array(
            [x.input_features_dict[input_feature_name].label_id for x in batch],
            np.int64))
    input_features_collated_dict[
        input_feature_name] = input_ids, mask, segment_ids, label_ids

  # include example ids for test submission
  try:
    example_ids = torch.tensor([int(x.example_id) for x in batch])
  except:
    example_ids = torch.zeros(len(batch)).long()

  if batch[0].bias is None:
    return example_ids, input_features_collated_dict

  teacher_probs = torch.tensor([x.teacher_probs for x in batch])
  bias = torch.tensor([x.bias for x in batch])

  return example_ids, input_features_collated_dict, bias, teacher_probs


def convert_examples_to_features(examples: List[TextPairExample],
                                 max_seq_length,
                                 tokenizer,
                                 n_process=1):
  converter = ExampleConverter(max_seq_length, tokenizer)
  return process_par(
      examples, converter, n_process, chunk_size=2000, desc="featurize")


class SortedBatchSampler(Sampler):

  def __init__(self, data_source, batch_size, seed):
    super().__init__(data_source)
    self.data_source = data_source
    self.batch_size = batch_size
    self.seed = seed
    if batch_size == 1:
      raise NotImplementedError()
    self._epoch = 0

  def __iter__(self):
    rng = np.random.RandomState(self._epoch + 601767 + self.seed)
    n_batches = len(self)
    batch_lens = np.full(n_batches, self.batch_size, np.int32)

    # Randomly select batches to reduce by size 1
    extra = n_batches * self.batch_size - len(self.data_source)
    batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

    batch_ends = np.cumsum(batch_lens)
    batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

    if batch_ends[-1] != len(self.data_source):
      print(batch_ends)
      raise RuntimeError()

    bounds = np.stack([batch_starts, batch_ends], 1)
    rng.shuffle(bounds)

    for s, e in bounds:
      yield np.arange(s, e)

  def __len__(self):
    return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
  if sorted:
    data.sort()
    ds = InputFeatureDataset(data)
    sampler = SortedBatchSampler(ds, batch_size, seed)
    return DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate_input_features)
  else:
    ds = InputFeatureDataset(data)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=RandomSampler(ds),
        collate_fn=collate_input_features)


def build_eval_dataloader(data: List[InputFeatures], batch_size):
  ds = InputFeatureDataset(data)
  return DataLoader(
      ds,
      batch_size=batch_size,
      sampler=SequentialSampler(ds),
      collate_fn=collate_input_features)

def simple_accuracy(preds, labels):
  return (preds == labels).mean()

def acc_and_f1(preds, labels):
  acc = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds)
  f1_non = f1_score(y_true=labels, y_pred=preds, pos_label=0)
  return {"acc": acc, "f1": f1, "f1_non": f1_non}


