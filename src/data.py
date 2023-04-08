import os
import random
from typing import Iterator

from torch import Tensor
from torch.utils.data.dataset import IterableDataset


class ShardedTextDataset(IterableDataset):
    def __init__(
        self,
        datadir: str,
        tokenizer,
        max_seq_length: int = 256,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        sort_by_length: bool = False
    ) -> IterableDataset:
        """
        A Dataset class for text data that has been split into a number of
        shards. Iteratively loads each shard into memory as it is used. Defines
        an iterable dataset API rather than a "map" (i.e. cannot use the
        __getitem__ method or indexing).

        Args:
            datadir: directory from which to read the data shards
            tokenizer: Huggingface text tokenizer
            max_seq_length: maximum number of tokens in a sequnece
            shuffle_shards: whether to shuffle the order of shards in one
                iteration (epoch)
            shuffle_within_shard: whether to shuffle examples/batches within
                each shard
            sort_by_length: whether to sort sequences by decreasing sequence
                length. Will save on memory allocated for padding if `True`
        """
        super().__init__()

        self.datadir = datadir
        self.datafiles = os.listdir(datadir)
        self.num_shards = len(self.datafiles)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.sort_by_length = sort_by_length

        # Calculate the total number of examples across an iteration (epoch).
        # Getting this number is useful for defining epoch length and/or
        # learning-rate schedules, but this operation involes reading all shards
        # and so should only be performed once
        self.total_num_examples = self._calculate_dataset_length()

    # Iterate over all shards and get the total number of examples
    def _calculate_dataset_length(self) -> int:
        total_num_examples = 0
        for datafile in self.datafiles:
            filepath = os.path.join(self.datadir, datafile)
            lines = [line for line in open(filepath, 'r') if line.strip() != '']
            total_num_examples += len(lines)
        return total_num_examples

    def __len__(self) -> int:
        return self.total_num_examples

    # Upon forming an iterator, set current_index to 0, then define an iteration
    # order over shards. Then use the _get_next_shard function to load the first
    # shard in
    def __iter__(self) -> Iterator[Tensor]:
        self.current_index = 0
        if self.shuffle_shards:
            datafiles_copy = self.datafiles.copy()
            random.shuffle(datafiles_copy)
            self.shard_iterator = iter(datafiles_copy)
        else:
            self.shard_iterator = iter(self.datafiles)
        self._get_next_shard()
        return self

    # Get the next batch of data from the current shard. Change over to the next
    # shard if the current one is completed
    def __next__(self) -> Tensor:
        if self.current_index < self.total_num_examples:
            if self.current_shard_index >= self.num_shard_examples:
                self._get_next_shard()
            current_example = self.current_shard_data[self.current_shard_index]
            self.current_index += 1
            self.current_shard_index += 1
            return self.tokenizer(
                current_example,
                max_length=self.max_seq_length,
                truncation=True
            )
        raise StopIteration

    def _get_next_shard(self):
        """
        Internal function to load the next shard from disk, optionally
        shuffling or sorting examples by length. Sets `self.current_shard_index`
        to 0.
        """
        self.current_shard_path = next(self.shard_iterator)
        self.current_shard_path = os.path.join(
            self.datadir, self.current_shard_path
        )
        shard_data = [
            line.strip()
            for line in open(self.current_shard_path, 'r') if line.strip() != ''
        ]
        self.num_shard_examples = len(shard_data)

        # Sort examples by length if desired. This is saves unneeded memory for
        # padding, but is undone if self.shuffle_within_shard is True
        if self.sort_by_length:
            shard_data.sort(key=len, reverse=True)
        if self.shuffle_within_shard:
            random.shuffle(shard_data)

        self.current_shard_data = shard_data
        self.current_shard_index = 0
