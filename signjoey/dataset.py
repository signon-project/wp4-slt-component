# coding: utf-8
"""
Data module
"""
from sklearn.preprocessing import normalize
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
from tqdm import tqdm   
import numpy as np
import pickle
import gzip
import torch
import json
import glob
import h5py

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        cfg,
        path: str,
        split,
        tokeniser,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        training=False,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
                ("txt_mask", fields[5]),
            ]

      
        langs = ["nl_XX", "en_XX", "es_XX"]
        annotations = ["annotation"]

        samples = {}
        with open(path, 'r') as f:
            samples = json.load(f)

        h5_spatial = h5py.File(cfg["embedding_file_spatial"], 'r')
        examples = []

        for sequence_id in tqdm(split):
            sample = samples[str(sequence_id)]
            video_id = sample["video_id"]
            embeddings = np.asarray(h5_spatial[str(sequence_id)])
       
            for i, k in enumerate(annotations): # can include more languages
                sentence = sample[k]
                tgt_lang = langs[i]
                inputs, mask = tokeniser.encode(sentence.strip(), add_special_tokens=True, tgt_lang=tgt_lang)
 
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["video_id"],
                            0,
                            torch.from_numpy(embeddings),
                            "",
                            inputs,
                            mask
                        ],
                        fields,
                    )
                )
            
        h5_spatial.close()
        super().__init__(examples, fields, **kwargs)