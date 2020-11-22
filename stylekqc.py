# coding=utf-8
# Extra tasks and data loaders for stylekqc
# Copyright 2020 Sangwhan Moon. All rights reserved.
#
# Original license:
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from __future__ import absolute_import, division, print_function

import csv
import os
import textwrap

import numpy as np
import six

import datasets


_GLUE_CITATION = """\
@inproceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
"""

_GLUE_DESCRIPTION = """\
GLUE, the General Language Understanding Evaluation benchmark
(https://gluebenchmark.com/) is a collection of resources for training,
evaluating, and analyzing natural language understanding systems.

"""


class StyleKQCConfig(datasets.BuilderConfig):
    """BuilderConfig for StyleKQC."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for StyleKQC.

        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          label_column: `string`, name of the column in the tsv file corresponding
            to the label
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(StyleKQCConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class StyleKQC(datasets.GeneratorBasedBuilder):
    """The StyleKQC dataset."""

    BUILDER_CONFIGS = [
        StyleKQCConfig(
            name="act",
            description=textwrap.dedent(
                """"""
            ),
            text_features={"sentence": "sentence"},
            label_classes=["0", "1", "2", "3"],
            label_column="act",
            data_url="",
            data_dir="act",
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        StyleKQCConfig(
            name="topic",
            description=textwrap.dedent(
                """"""
            ),
            text_features={"sentence": "sentence"},
            label_classes=["0", "1", "2", "3", "4", "5"],
            label_column="topic",
            data_url="",
            data_dir="topic",
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        StyleKQCConfig(
            name="sts",
            description=textwrap.dedent(
                """"""
            ),
            text_features={
                "sentence1": "sentence1",
                "sentence2": "sentence2",
            },
            label_classes=["dissimilar", "similar"],
            label_column="similarity",
            data_url=None,
            data_dir="sts",
            citation=textwrap.dedent(
                """"""
            ),
            url="",
            process_label=np.float32,
        ),
    ]

    def _info(self):
        features = {text_feature: datasets.Value("string") for text_feature in six.iterkeys(self.config.text_features)}
        if self.config.label_classes:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        else:
            features["label"] = datasets.Value("float32")
        features["idx"] = datasets.Value("int32")
        return datasets.DatasetInfo(
            description=_GLUE_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _GLUE_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "ax":
            data_file = dl_manager.download(self.config.data_url)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": data_file,
                        "split": "test",
                    },
                )
            ]

        data_dir = self.config.data_dir
        train_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "data_file": os.path.join(data_dir or "", "train.tsv"),
                "split": "train"
            },
        )
        return [
            train_split,
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", "dev.tsv"),
                    "split": "dev"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", "test.tsv"),
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, data_file, split, mrpc_files=None):
        process_label = self.config.process_label
        label_classes = self.config.label_classes

        with open(data_file, encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            for n, row in enumerate(reader):

                example = {feat: row[col] for feat, col in six.iteritems(self.config.text_features)}
                example["idx"] = n

                if self.config.label_column in row:
                    label = row[self.config.label_column]
                    # For some tasks, the label is represented as 0 and 1 in the tsv
                    # files and needs to be cast to integer to work with the feature.
                    if label_classes and label not in label_classes:
                        label = int(label) if label else None
                    example["label"] = process_label(label)
                else:
                    example["label"] = process_label(-1)

                # Filter out corrupted rows.
                for value in six.itervalues(example):
                    if value is None:
                        break
                else:
                    yield example["idx"], example
