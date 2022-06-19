"""
Utils for train*.py.
"""
import random
import string
from typing import Dict, Iterable, List
from functools import total_ordering
from collections import namedtuple
from utils import Processor, process_par
import numpy as np
import math

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler

from sklearn.metrics import f1_score

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
  SHUFFLED_INPUT_LIST = "shuffled_input_feature_list"
  TOKEN_DROPOUT_INPUTS_LIST = "token_dropout_input_feature_list"

  def __init__(self, example_id, original_input, shuffled_input_list,
               token_dropout_input_list, bias):
    self.example_id = example_id
    self.input_features_dict = {
        self.ORIGINAL_INPUT: original_input,
        self.SHUFFLED_INPUT_LIST: shuffled_input_list,
        self.TOKEN_DROPOUT_INPUTS_LIST: token_dropout_input_list
    }
    self.bias = bias
    self.max_sequence_length = self._compute_max_seq_len()

  def _get_key(self):
    """Preserves the original code's key used for sorting this class."""
    return self.input_features_dict.get(self.ORIGINAL_INPUT).input_ids

  def __eq__(self, other):
    return self._get_key() == other._get_key()

  def __lt__(self, other):
    return self._get_key() < other._get_key()

  def get_original_label_id(self):
    return self.input_features_dict[self.ORIGINAL_INPUT].label_id
  
  def _compute_max_seq_len(self):
    max_len = 0
    for x in self.input_features_dict.values():
      if isinstance(x, InputFeature):
        max_len = max(max_len, len(x.input_ids))
      else:
        # Must be a list of InputFeature objects.
        max_len = max(max_len, max([len(y.input_ids) for y in x]))
    return max_len
    
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

  def __init__(self, max_seq_length, tokenizer, num_of_aug_generated=5):
    self.max_seq_length = max_seq_length
    self.tokenizer = tokenizer
    self.num_of_aug_generated = num_of_aug_generated

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

  def _get_input_feature(self, premise, hypothesis, label_id):
    input_ids, segment_ids = self._get_input_and_segment_ids(
        premise, hypothesis)
    return InputFeature(
        input_ids=input_ids, segment_ids=segment_ids, label_id=label_id)

  @staticmethod
  def _shuffle_sentence_like_word_ordering_paper(text_pair_example,
                                                 random_seed):
    sentence = text_pair_example.hypothesis
    if len(nltk.sent_tokenize(sentence)) > 1:
      return None
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) < 3:
      return None
    random.Random(random_seed).shuffle(tokens)
    return " ".join(tokens)

  @staticmethod
  def _shuffle_hyp_but_keep_mutual_phrases(text_pair_example, random_seed):
    new_hypothesis, _ = shuffle_but_keep_mutual_phrases_together(
        text_pair_example.hypothesis, text_pair_example.premise, random_seed)
    return new_hypothesis

  @staticmethod
  def _shuffle_premise_but_keep_mutual_phrases(text_pair_example, random_seed):
    _, new_premise = shuffle_but_keep_mutual_phrases_together(
        text_pair_example.hypothesis, text_pair_example.premise, random_seed)
    return new_premise

  @staticmethod
  def _identity(text_pair_example, random_seed):
    "Rely on the bert model's dropout during training."
    return text_pair_example.hypothesis

  @staticmethod
  def _dropout_tokens(text_pair_example, random_seed, dropout_prob=0.10):
    sentence = text_pair_example.hypothesis
    tokens = nltk.word_tokenize(sentence)
    num_tokens_to_keep = math.floor((1 - dropout_prob) * len(tokens))
    indicies_to_keep = random.Random(random_seed).sample(
        range(len(tokens)), num_tokens_to_keep)
    tokens_to_keep = [tokens[i] for i in sorted(indicies_to_keep)]
    return " ".join(tokens_to_keep)

  def get_aug_input_features_list(self,
                                  text_pair_example,
                                  aug_fn,
                                  num_to_generate=None):
    if num_to_generate is None:
      num_to_generate = self.num_of_aug_generated
    augmented_input_feature_list = []
    for i in range(num_to_generate):
      text_pair_example = aug_fn(text_pair_example, i)
      input_feature = self._get_input_feature(text_pair_example.premise,
                                              text_pair_example.hypothesis,
                                              text_pair_example.label)
      augmented_input_feature_list.append(input_feature)
    return augmented_input_feature_list

  def create_input_features(self, text_pair_example):
    original_input_feature = self._get_input_feature(
        text_pair_example.premise, text_pair_example.hypothesis,
        text_pair_example.label)

    def get_aughyp_input_features_list(hypothesis_aug_fn, num_to_generate=None):
      def aug_fn(text_pair_example, i):
        aug_hypothesis = hypothesis_aug_fn(text_pair_example.hypothesis, i)
        TextPairExample(text_pair_example.id, text_pair_example.premise,
                        aug_hypothesis, text_pair_example.label)
      return self.get_aug_input_features_list(text_pair_example, aug_fn,
                                              num_to_generate)

    def get_augprem_input_features_list(premise_aug_fn, num_to_generate=None):
      def aug_fn(text_pair_example, i):
        aug_premise = premise_aug_fn(text_pair_example.premise, i)
        TextPairExample(text_pair_example.id, aug_premise,
                        text_pair_example.hypothesis, text_pair_example.label)
      return self.get_aug_input_features_list(text_pair_example, aug_fn,
                                              num_to_generate)

    half = self.num_of_aug_generated // 2
    other_half = self.num_of_aug_generated - half
    shuffled_input_feature_list = get_aughyp_input_features_list(
        ExampleConverter._shuffle_hyp_but_keep_mutual_phrases,
        half) + get_augprem_input_features_list(
            ExampleConverter._shuffle_prem_but_keep_mutual_phrases, other_half)

    token_dropout_input_feature_list = get_aughyp_input_features_list(
        ExampleConverter._identity, 1)

    return InputFeatures(
        example_id=text_pair_example.id,
        original_input=original_input_feature,
        shuffled_input_list=shuffled_input_feature_list,
        token_dropout_input_list=token_dropout_input_feature_list,
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

  max_seq_len = max(x.max_sequence_length for x in batch)

  input_feature_name_to_count = {}
  for input_feature_name in input_feature_names:
    if input_feature_name == InputFeatures.ORIGINAL_INPUT:
      pass
    count = min(len(x.input_features_dict[input_feature_name]) for x in batch)
    input_feature_name_to_count[input_feature_name] = count

  def collate_input_feature(get_input_feature_fn):
    """Fn passed in gets a InputFeature from InputFeatures object."""
    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
      input_feature = get_input_feature_fn(ex)
      input_ids[i, :len(input_feature.input_ids)] = input_feature.input_ids
      segment_ids[
          i, :len(input_feature.segment_ids)] = input_feature.segment_ids
      mask[i, :len(input_feature.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(
        np.array([get_input_feature_fn(x).label_id for x in batch], np.int64))
    return input_ids, mask, segment_ids, label_ids

  input_features_collated_dict = {}
  for ifn in input_feature_names:
    if ifn == InputFeatures.ORIGINAL_INPUT:
      fn = lambda ex: ex.input_features_dict[ifn]
      input_ids, mask, segment_ids, label_ids = collate_input_feature(fn)
      input_features_collated_dict[
          ifn] = input_ids, mask, segment_ids, label_ids
    else:
      values = []
      for i in range(input_feature_name_to_count[ifn]):
        fn = lambda ex: ex.input_features_dict[ifn][i]
        input_ids, mask, segment_ids, label_ids = collate_input_feature(fn)
        values.append((input_ids, mask, segment_ids, label_ids))
      input_features_collated_dict[ifn] = values

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


def build_train_dataloader(data: List[InputFeatures], batch_size, seed,
                           do_sort):
  if do_sort:
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

# Other augmentation functions.
def word_tokenize_with_index(text):
  tokens=nltk.word_tokenize(text)
  offset = 0
  result = []
  for token in tokens:
    offset = text.find(token, offset)
    result.append((token.lower(), offset, offset+len(token)))
    offset += len(token)
  if result[-1][0] in string.punctuation:
    end = result.pop()
    return result, end
  return result, None

def shuffle_but_keep_mutual_phrases_together(s1, s2, seed=0):
  """
  Given two strings {s1} and {s2}, it returns a shuffling of their tokens:
  - tokens are determined by nltk.word_tokenize
  - If there is a common phrase of multiple tokens across the two
    sentences, that phrase is kept together during the shuffle. What
    constitutes a mutual phrase is case insenstive. The mutual phrases are
    shuffled 50 times or until the phrases are not in their initial ordering.
    This is necessary, since sometimes a sentence only has two mutual phrases
    and we have to break intelligibility. 
    This sometimes means that the shuffled sentence is larger than the original 
    if the same token is part of two mutual phrases:
       ex: The scientist ate the cake
           The scientist ate the onion or ate the cake.
           Here the mutual phrases are:
              "The scientist ate the", and "the cake"
           The current behavior includes both phrases, even though this does
           increase the overall length of the sentence.
  - If either sentence ends in a token present in string.punctuation, 
    that token continues to be at the end of that sentence.
  """
  s1_tokens, s1_end = word_tokenize_with_index(s1)
  s2_tokens, s2_end = word_tokenize_with_index(s2)

  # Get substrings in common using token-wise index.
  subtring_end_index_inclusive_to_len = {}
  for i, (s1t, _, _) in enumerate(s1_tokens):
    for j, (s2t, _, _) in enumerate(s2_tokens):
      if s1t == s2t:
        prev = subtring_end_index_inclusive_to_len.get((i-1, j-1), 0)
        subtring_end_index_inclusive_to_len[(i, j)] = prev + 1
        if prev != 0:
          del subtring_end_index_inclusive_to_len[(i - 1, j - 1)]  # Save space.

  def is_i_subset_of_j(i_start, i_length, j_start, j_length):
    if j_start > i_start:  # second item starts strictly after first item.
      return False
    i_end = i_start + i_length
    j_end = j_start + j_length
    return j_end >= i_end  # second item ends strictly at or after first item.

  def filter_subset_substrings(start_index_and_len):
    # Sort length increasing, and then start_index increasing.
    start_index_and_len.sort(key=lambda x: (x[1], x[0]))
    min_start_index_inclusive_and_len = []
    for i, (i_start, i_length) in enumerate(start_index_and_len):
      next_items = start_index_and_len[i + 1:]
      is_i_subset = any([
          is_i_subset_of_j(i_start, i_length, j_start, j_length)
          for (j_start, j_length) in next_items
      ])
      if not is_i_subset:
        min_start_index_inclusive_and_len.append((i_start, i_length))
    return min_start_index_inclusive_and_len

  s1_start_index_and_len = filter_subset_substrings([
      (e1 - length + 1, length)
      for ((e1, _), length) in subtring_end_index_inclusive_to_len.items()
  ])
  s2_start_index_and_len = filter_subset_substrings([
      (e2 - length + 1, length)
      for ((_, e2), length) in subtring_end_index_inclusive_to_len.items()
  ])

  def get_substrings(text, tokens, start_index_and_len):
    get_substring = lambda token_index_range: text[tokens[token_index_range[0]][
        1]:tokens[token_index_range[1]][2]]
    i = 0
    substrings = []
    is_phrase = []
    for s, length in sorted(start_index_and_len):
      for j in range(i, s):
        substrings.append(get_substring((j, j)))
        is_phrase.append(False)
      i = s + length
      substrings.append(get_substring((s, i - 1)))
      is_phrase.append(length > 1)
    for j in range(i, len(tokens)):
      substrings.append(get_substring((j, j)))
      is_phrase.append(False)
    return substrings, is_phrase

  s1_substrings, s1_is_phrase = list(
      get_substrings(s1, s1_tokens, s1_start_index_and_len))
  s2_substrings, s2_is_phrase = list(
      get_substrings(s2, s2_tokens, s2_start_index_and_len))

  is_sorted = lambda l: (all(l[i] <= l[i + 1] for i in range(len(l) - 1)))

  def shuffle_and_join(text, substrings, substring_is_phrase, end=None):
    phrase_indicies = set(i for i, b in enumerate(substring_is_phrase) if b)
    substring_and_indices = list(enumerate(substrings))

    for salt in range(50):  # Try 50 times to make sure phrases are shuffled.
      random.Random(seed+salt).shuffle(substring_and_indices)
      phrase_shuffled_indicies = [
          x[0] for x in substring_and_indices if x[0] in phrase_indicies
      ]
      if not is_sorted(phrase_shuffled_indicies):
        break
    substrings = [x[1] for x in substring_and_indices]
    new_text = " ".join(substrings)
    if end is not None:
      new_text += text[end[1]:end[2]]
    return new_text
  new_s1 = shuffle_and_join(s1, s1_substrings, s1_is_phrase, s1_end)
  new_s2 = shuffle_and_join(s2, s2_substrings, s2_is_phrase, s2_end)

  return new_s1, new_s2

