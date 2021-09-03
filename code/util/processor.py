# coding=utf-8

"""Processors for different tasks."""

import csv
import os
import json

import pandas as pd
import pickle

from util.tokenization import *

import re
import sys

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
class SST2_Processor(DataProcessor):
    """Processor for the SST data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(test_data, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(test_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1] # 0: very negative ->  4: very positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(line[1])
            if i == 0 and debug:
                print("*** Example ***")
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples