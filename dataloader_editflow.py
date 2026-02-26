import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import torch.nn.functional as F
import transformers

import utils
from flow_utils import opt_align_xs_to_zs
from flows import EmptyCoupling, UniformCoupling

os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
LOGGER = utils.get_logger(__name__)


def cnn_dm_detokenizer(x):
    # Remove common news wire prefixes
    x = re.sub(r"^.*?\(CNN\) -- ", "", x)
    x = re.sub(r"^.*?\(CNN\)", "", x)
    x = x.replace(" -- ", "—")  # Standardize dashes

    # Standard cleanup (similar to lm1b but less aggressive on punctuation)
    x = x.replace(" , ", ", ")
    x = x.replace(" . ", ". ")
    x = x.replace(" ! ", "! ")
    x = x.replace(" ? ", "? ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" 's", "'s")
    x = x.replace(" ' ", "'")

    # Fix whitespace
    x = x.strip()
    return x


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n' + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
            self,
            bos_token='[BOS]',
            eos_token='[EOS]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            pad_token='[PAD]',
            mask_token='[MASK]',
            unk_token='[UNK]',
            **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz ')
        self._vocab_str_to_int = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[PAD]': 5,
            '[RESERVED]': 6,
            '[UNK]': 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)}}
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
    """Adapted from:
      https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

      Args:
        cache_dir: str, path to cache directory.
        max_seq_length: int, maximum length of sequences.
            (default: 256, as in D3PM codebase.)
        drop_last: bool, whether to drop the last incomplete
            batch. (default: True, as in D3PM codebase.)
        crop_train: bool, whether to subsample contiguous
            subsequences from training example. serves to
            make sure transformer models with absolute position
            embeddings do not have incorrect position-wise
            marginals. (default: False, but necessary to match D3PM AR)

      Returns:
        dataset: dataset.DatasetDict, with keys 'train',
            'valid', 'test'.
    """
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not crop_train:
        cache_dir = f'{cache_dir}/text8'
    else:
        cache_dir = f'{cache_dir}/text8-crop-train'
    split_names = ['train', 'validation', 'test']
    if not all([
        utils.fsspec_exists(os.path.join(cache_dir, split))
        for split in split_names
    ]):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, 'raw_data')
        if not all([
            utils.fsspec_exists(
                os.path.join(raw_cache_dir, f'text8.{split}.txt'))
            for split in split_names
        ]):
            if not utils.fsspec_exists(
                    os.path.join(raw_cache_dir, 'text8.zip')):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info('Downloading text8 from URL {}.'.format(url))
                with (urllib.request.urlopen(url) as in_stream,
                      open(os.path.join(raw_cache_dir, 'text8.zip'),
                           'wb') as out_file):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(
                    os.path.join(raw_cache_dir, 'text8.zip'),
                    'rb') as f:
                rawdata = zipfile.ZipFile(f).read(
                    'text8').decode('utf-8')

            # Splits taken from D3PM codebase
            splits = {
                'train': rawdata[:90000000],
                'validation': rawdata[90000000: 95000000],
                'test': rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir,
                                     f'text8.{split}.txt')
                with fsspec.open(_path, 'w') as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir,
                                     f'text8.{split}.txt')
                with fsspec.open(_path, 'r') as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == 'train' and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples['input_ids']))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos]
            + concatenated_examples[i: i + new_block_size]
            + [eos])
        _attn_masks.append(torch.ones(block_size))
    result['input_ids'] = _values
    result['attention_mask'] = _attn_masks
    return result


def get_dataset(
        dataset_name, tokenizer, wrap, mode, cache_dir,
        block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):

    print (streaming)
    if not streaming:
        if wrap:
            filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
        else:
            filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
        _path = os.path.join(cache_dir, filename)

        if utils.fsspec_exists(_path):
            LOGGER.info(f'Loading data from: {_path}')
            return datasets.load_from_disk(_path).with_format('torch')
        LOGGER.info(f'Generating new data at: {_path}')
    else:
        LOGGER.info(f'Streaming data for: {dataset_name} (No local cache)')
        _path = None  # No path for streaming

    crop_train = dataset_name == 'text8-crop'
    if mode == 'train' and crop_train:
        block_size *= 2

    # --- DATASET LOADING LOGIC ---
    if dataset_name == 'wikitext103':
        dataset = datasets.load_dataset(
            'wikitext', name='wikitext-103-raw-v1', cache_dir=cache_dir)

    elif dataset_name == 'c4_realnewslike':
        dataset = datasets.load_dataset(
            'allenai/c4', 'realnewslike',
            cache_dir=cache_dir,
            streaming=streaming
        )



    elif dataset_name == 'wikitext2':
        dataset = datasets.load_dataset(
            'wikitext', name='wikitext-2-raw-v1', cache_dir=cache_dir)
    elif dataset_name == 'ptb':
        dataset = datasets.load_dataset('ptb_text_only', cache_dir=cache_dir)
    elif dataset_name == 'lambada':
        dataset = get_lambada_test_dataset()
    elif dataset_name == 'text8':
        assert wrap
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == 'text8-crop':
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size, crop_train=True)
    elif dataset_name == 'cnn_dailymail':
        # Load version 3.0.0 (non-anonymized)
        dataset = datasets.load_dataset(
            'cnn_dailymail', '3.0.0',
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'openwebtext-train':
        dataset = datasets.load_dataset(
            'openwebtext', split='train[:-100000]', cache_dir=cache_dir, streaming=streaming)
    elif dataset_name == 'openwebtext-valid':
        dataset = datasets.load_dataset(
            'openwebtext', split='train[-100000:]', cache_dir=cache_dir, streaming=streaming)
    elif dataset_name.startswith('scientific_papers'):
        sub = dataset_name.split('_')[-1]  # arxiv or pubmed
        dataset = datasets.load_dataset(
            'scientific_papers', sub, trust_remote_code=True, cache_dir=cache_dir, streaming=streaming)
    elif dataset_name == 'fineweb':
        # The new state-of-the-art web dataset (replaces C4)
        dataset = datasets.load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-100BT",  # or "default" for full 15TB
            split="train",
            streaming=streaming
        )
    elif dataset_name == 'ag_news':
        dataset = datasets.load_dataset('ag_news', cache_dir=cache_dir, streaming=streaming)
    else:
        dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir, streaming=streaming)

    if dataset_name == 'c4_realnewslike':
        if mode == 'test':
            data = dataset['validation']
        else:
            data = dataset[mode]
    elif dataset_name in ['lambada', 'openwebtext-train', 'openwebtext-valid', 'fineweb']:
        data = dataset
    else:
        data = dataset[mode]

    # --- DETOKENIZER SELECTION ---
    if dataset_name.startswith('wikitext'):
        detokenizer = wt_detokenizer
    elif dataset_name == 'c4_realnewslike':
        # Re-use cnn_dm_detokenizer or a simple whitespace fixer
        detokenizer = lambda x: x.strip()


    elif dataset_name == 'ptb':
        detokenizer = ptb_detokenizer
    elif dataset_name == 'lm1b':
        detokenizer = lm1b_detokenizer
    elif dataset_name == 'lambada':
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith('scientific_papers'):
        detokenizer = scientific_papers_detokenizer
    elif dataset_name == 'cnn_dailymail':
        detokenizer = cnn_dm_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detok_fn):
        def detok(text_list):
            # Optimized to avoid loop overhead if possible, but list comp is fine
            return [detok_fn(t) for t in text_list]

        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    # --- TOKENIZATION ---
    def preprocess_and_tokenize(example):
        # FIELD SELECTION
        if dataset_name == 'ptb':
            text = example['sentence']
        elif 'scientific_papers' in dataset_name:
            text = example['article']
        elif dataset_name == 'cnn_dailymail':
            text = example['article']  # Explicitly select article, ignore highlights
        else:
            text = example['text']

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        if wrap:
            tokens = tokenizer(text,
                               add_special_tokens=False,
                               return_attention_mask=False,
                               return_token_type_ids=False)
            tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
        else:
            tokens = tokenizer(text,
                               max_length=block_size,
                               padding='max_length',
                               truncation=True,
                               add_special_tokens=True,
                               return_attention_mask=True,
                               return_token_type_ids=True)
        return tokens

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize, batched=True) #, desc='Tokenizing')
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize, batched=True, num_proc=num_proc,
            load_from_cache_file=True, desc='Tokenizing')

    # --- COLUMN CLEANUP ---
    if dataset_name == 'ptb':
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')

    elif 'scientific_papers' in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns(['article', 'abstract', 'section_names'])
    elif dataset_name == 'cnn_dailymail':
        # Remove CNN specific columns
        tokenized_dataset = tokenized_dataset.remove_columns(['article', 'highlights', 'id'])
    elif dataset_name == 'ag_news':
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')

    if not wrap:
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format('torch')

    group_texts = functools.partial(_group_texts, block_size=block_size, bos=BOS, eos=EOS)

    if streaming:
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True, desc='Grouping')
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts, batched=True, num_proc=num_proc,
            load_from_cache_file=True, desc='Grouping')
        chunked_dataset.save_to_disk(_path)

    chunked_dataset = chunked_dataset.with_format('torch')
    return chunked_dataset


# def get_dataset(
#         dataset_name, tokenizer, wrap, mode, cache_dir,
#         block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):
#     if wrap:
#         filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
#     else:
#         filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
#     _path = os.path.join(cache_dir, filename)
#
#     if utils.fsspec_exists(_path):
#         LOGGER.info(f'Loading data from: {_path}')
#         return datasets.load_from_disk(_path).with_format('torch')
#     LOGGER.info(f'Generating new data at: {_path}')
#
#     crop_train = dataset_name == 'text8-crop'
#     if mode == 'train' and crop_train:
#         # double block size for sub-sampling
#         block_size *= 2
#
#     if dataset_name == 'wikitext103':
#         dataset = datasets.load_dataset(
#             'wikitext',
#             name='wikitext-103-raw-v1',
#             cache_dir=cache_dir)
#     elif dataset_name == 'wikitext2':
#         dataset = datasets.load_dataset(
#             'wikitext',
#             name='wikitext-2-raw-v1',
#             cache_dir=cache_dir)
#     elif dataset_name == 'ptb':
#         dataset = datasets.load_dataset(
#             'ptb_text_only', cache_dir=cache_dir)
#     elif dataset_name == 'lambada':
#         dataset = get_lambada_test_dataset()
#     elif dataset_name == 'text8':
#         assert wrap
#         dataset = get_text8_dataset(
#             cache_dir, max_seq_length=block_size)
#     elif dataset_name == 'text8-crop':
#         dataset = get_text8_dataset(
#             cache_dir, max_seq_length=block_size, crop_train=True)
#     elif dataset_name == 'openwebtext-train':
#         dataset = datasets.load_dataset(
#             'openwebtext',
#             split='train[:-100000]',
#             cache_dir=cache_dir,
#             streaming=streaming)
#     elif dataset_name == 'openwebtext-valid':
#         dataset = datasets.load_dataset(
#             'openwebtext',
#             split='train[-100000:]',
#             cache_dir=cache_dir,
#             streaming=streaming)
#     elif dataset_name == 'scientific_papers_arxiv':
#         dataset = datasets.load_dataset(
#             'scientific_papers', 'arxiv',
#             trust_remote_code=True,
#             cache_dir=cache_dir,
#             streaming=streaming)
#     elif dataset_name == 'scientific_papers_pubmed':
#         dataset = datasets.load_dataset(
#             'scientific_papers', 'pubmed',
#             trust_remote_code=True,
#             cache_dir=cache_dir,
#             streaming=streaming)
#     elif dataset_name == 'ag_news':
#         dataset = datasets.load_dataset(
#             'ag_news',
#             cache_dir=cache_dir,
#             streaming=streaming)
#     else:
#         dataset = datasets.load_dataset(
#             dataset_name,
#             cache_dir=cache_dir,
#             streaming=streaming)
#
#     if dataset_name in ['lambada', 'openwebtext-train',
#                         'openwebtext-valid']:
#         data = dataset
#     else:
#         data = dataset[mode]
#
#     if dataset_name.startswith('wikitext'):
#         detokenizer = wt_detokenizer
#     elif dataset_name == 'ptb':
#         detokenizer = ptb_detokenizer
#     elif dataset_name == 'lm1b':
#         detokenizer = lm1b_detokenizer
#     elif dataset_name == 'lambada':
#         detokenizer = lambada_detokenizer
#     elif dataset_name.startswith('scientific_papers'):
#         detokenizer = scientific_papers_detokenizer
#     else:
#         detokenizer = None
#
#     def _apply_detokenizer(detokenizer):
#         def detok(text):
#             for i, t in enumerate(text, 0):
#                 text[i] = detokenizer(t)
#             return text
#
#         return detok
#
#     EOS = tokenizer.encode(tokenizer.eos_token)[0]
#     BOS = tokenizer.encode(tokenizer.bos_token)[0]
#
#     def preprocess_and_tokenize(example):
#         if dataset_name == 'ptb':
#             text = example['sentence']
#         elif 'scientific_papers' in dataset_name:
#             text = example['article']
#         else:
#             text = example['text']
#
#         if detokenizer is not None:
#             text = _apply_detokenizer(detokenizer)(text)
#
#         tokenizer.padding_side = 'right'
#         tokenizer.truncation_side = 'right'
#
#         if wrap:
#             tokens = tokenizer(text,
#                                add_special_tokens=False,
#                                return_attention_mask=False,
#                                return_token_type_ids=False)
#             tokens = {'input_ids':
#                           [t + [EOS] for t in tokens['input_ids']]}
#             # Still missing BOS, but will be added in group_texts
#         else:
#             tokens = tokenizer(text,
#                                max_length=block_size,
#                                padding='max_length',
#                                truncation=True,
#                                add_special_tokens=True,
#                                return_attention_mask=True,
#                                return_token_type_ids=True)
#         return tokens
#
#     if streaming:
#         tokenized_dataset = data.map(
#             preprocess_and_tokenize,
#             batched=True,
#             desc='Tokenizing')
#     else:
#         tokenized_dataset = data.map(
#             preprocess_and_tokenize,
#             batched=True,
#             num_proc=num_proc,
#             load_from_cache_file=True,
#             desc='Tokenizing')
#     if dataset_name == 'ptb':
#         tokenized_dataset = tokenized_dataset.remove_columns(
#             'sentence')
#     elif 'scientific_papers' in dataset_name:
#         tokenized_dataset = tokenized_dataset.remove_columns([
#             'article', 'abstract', 'section_names'])
#     elif dataset_name == 'ag_news':
#         tokenized_dataset = tokenized_dataset.remove_columns(
#             ['text', 'label'])
#     else:
#         tokenized_dataset = tokenized_dataset.remove_columns(
#             'text')
#
#     if not wrap:
#         tokenized_dataset.save_to_disk(_path)
#         return tokenized_dataset.with_format('torch')
#
#     group_texts = functools.partial(
#         _group_texts, block_size=block_size, bos=BOS, eos=EOS)
#     if streaming:
#         chunked_dataset = tokenized_dataset.map(
#             group_texts,
#             batched=True,
#             desc='Grouping')
#     else:
#         chunked_dataset = tokenized_dataset.map(
#             group_texts,
#             batched=True,
#             num_proc=num_proc,
#             load_from_cache_file=True,
#             desc='Grouping')
#         chunked_dataset.save_to_disk(_path)
#     chunked_dataset = chunked_dataset.with_format('torch')
#     return chunked_dataset
#

def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == 'text8':
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer. \
            from_pretrained('bert-base-uncased')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path)

    if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
            or isinstance(tokenizer, transformers.GPT2Tokenizer)):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id))

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def collate_edit_batch(batch, coupling, seq_align_fn, pad_token, vocab_size, gap_token_id, del_prob=0.05):
    x1_list = [b['input_ids'] for b in batch]

    x_1, x_0 = [], []
    z_1, z_0 = [], []


    t = torch.rand(len(x1_list), 1)
    t = torch.clamp(t, min=0.00, max=0.99)

    for i, _x1 in enumerate(x1_list):
        # 1. Clean Inputs
        _x1 = _x1[_x1 != pad_token]
        _x1 = _x1.unsqueeze(0)

        # 2. Sample & Align
        # Start with standard alignment (usually empty/masked source)
        _x0, _ = coupling.sample(_x1)
        _z0, _z1 = seq_align_fn(_x0, _x1)

        _z0_flat = _z0.squeeze(0)
        _z1_flat = _z1.squeeze(0)

        # --- PHASE A: DELETION NOISE ONLY ---
        # Goal: Insert [Garbage] in z0 and [Gap] in z1 at the same positions.
        seq_len = _z0_flat.size(0)

        # We can insert before any existing token or at the end
        n_slots = seq_len + 1
        ins_mask = torch.rand(n_slots) < del_prob

        # ensure only a maximum number are allowed:
            # Limit the number of insertions to prevent sequence explosion


        if ins_mask.any() and t[i] > 0.8:
            num_ins = ins_mask.sum().item()

            max_allowed_ins = int(seq_len * del_prob)  # Allow up to del_prob
            if num_ins > max_allowed_ins:
                # Randomly select a subset of insertion positions
                ins_indices_flat = torch.nonzero(ins_mask).squeeze(1)
                perm = torch.randperm(len(ins_indices_flat), device=_x0.device)
                selected_ins_indices = ins_indices_flat[perm[:max_allowed_ins]]

                ins_mask = torch.zeros_like(ins_mask, dtype=torch.bool)
                ins_mask[selected_ins_indices] = True
                num_ins = ins_mask.sum().item()

            # 1. Generate Noise for Source (The "Error" to delete)
            # We sample from full vocab, then resample any collisions with Pad/Gap
            noise_tokens = torch.randint(0, vocab_size, (num_ins,), device=_x0.device)

            # Rejection sampling loop to ensure no Pad or Gap tokens are picked
            mask_invalid = (noise_tokens == pad_token) | (noise_tokens == gap_token_id)

            while mask_invalid.any():
                # Resample only the invalid ones
                num_invalid = mask_invalid.sum().item()
                new_tokens = torch.randint(0, vocab_size, (num_invalid,), device=_x0.device)
                noise_tokens[mask_invalid] = new_tokens

                # Re-check
                mask_invalid = (noise_tokens == pad_token) | (noise_tokens == gap_token_id)

            # 2. Generate Gaps for Target (The "Correct" empty state)
            gap_tokens = torch.full((num_ins,), gap_token_id, dtype=torch.long, device=_x0.device)

            # 3. Interleave logic to insert new rows
            z0_parts, z1_parts = [], []
            ins_indices = torch.nonzero(ins_mask).squeeze(1).tolist()

            last_pos = 0
            noise_idx = 0

            for pos in ins_indices:
                # Copy existing valid chunk
                if pos > last_pos:
                    z0_parts.append(_z0_flat[last_pos:pos])
                    z1_parts.append(_z1_flat[last_pos:pos])

                # Insert the Noise/Gap pair
                z0_parts.append(noise_tokens[noise_idx: noise_idx + 1])
                z1_parts.append(gap_tokens[noise_idx: noise_idx + 1])
                noise_idx += 1

                last_pos = pos

            # Append remainder
            if last_pos < seq_len:
                z0_parts.append(_z0_flat[last_pos:])
                z1_parts.append(_z1_flat[last_pos:])

            _z0_flat = torch.cat(z0_parts)
            _z1_flat = torch.cat(z1_parts)

        # Reshape back to [1, Seq]
        _z0 = _z0_flat.unsqueeze(0)
        _z1 = _z1_flat.unsqueeze(0)

        # Reconstruct x0 from the modified z0 (removing only gaps)
        # Any garbage tokens we added are NOT gaps, so they stay in x0
        _x0_rec = _z0[_z0 != gap_token_id].unsqueeze(0)

        x_1.append(_x1.squeeze(0))
        x_0.append(_x0_rec.squeeze(0))
        z_1.append(_z1.squeeze(0))
        z_0.append(_z0.squeeze(0))

    # 3. Padding (Standard Logic)
    x0_max_len = max(len(x) for x in x_0)
    x1_max_len = max(len(x) for x in x_1)
    z_max_len = max(len(z) for z in z_1)

    x_1 = torch.stack([F.pad(x, (0, x1_max_len - x.shape[0]), value=pad_token) for x in x_1], dim=0).long()
    x_0 = torch.stack([F.pad(x, (0, x0_max_len - x.shape[0]), value=pad_token) for x in x_0], dim=0).long()
    z_1 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_1], dim=0).long()
    z_0 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_0], dim=0).long()


    return x_0, x_1, z_0, z_1, t


#
# def collate_edit_batch(batch, coupling, seq_align_fn, pad_token):
#     # print (batch)
#     # exit()
#     x1_list = [b['input_ids'] for b in batch]
#
#     x_1, x_0 = [], []
#     z_1, z_0 = [], []
#
#
#
#     for _x1 in x1_list:
#         # seq_len = torch.sum(_x1 != )
#         _x1 = _x1[_x1 != pad_token]
#         _x1 = _x1.unsqueeze(0)
#         _x0, _ = coupling.sample(_x1)
#         _z0, _z1 = seq_align_fn(_x0, _x1)
#         # print (_x1, _x0, _z1, _z0, _x1.shape, _x0.shape, _z1.shape, _z0.shape)
#         x_1.append(_x1.squeeze(0))
#         x_0.append(_x0.squeeze(0))
#         z_1.append(_z1.squeeze(0))
#         z_0.append(_z0.squeeze(0))
#
#     # Find the maximum length of each sequence in the batch
#     x0_max_len = max(len(x) for x in x_0)
#     x1_max_len = max(len(x) for x in x_1)
#     z_max_len = max(len(z) for z in z_1)
#     assert z_max_len == max(len(z) for z in z_0), "z_1 and z_0 must have the same max length"
#
#     # Add <PAD> token at end of each sequence to make them equal length
#     x_1 = torch.stack([F.pad(x, (0, x1_max_len - x.shape[0]), value=pad_token) for x in x_1], dim=0).long()
#     x_0 = torch.stack([F.pad(x, (0, x0_max_len - x.shape[0]), value=pad_token) for x in x_0], dim=0).long()
#     z_1 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_1], dim=0).long()
#     z_0 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_0], dim=0).long()
#
#     t = torch.rand(x_1.shape[0], 1)
#
#     t = torch.clamp(t, min=0.01, max=0.99)
#
#
#     return x_0, x_1, z_0, z_1, t
#
def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
    num_gpus = torch.cuda.device_count()

    # # ... (Keep your existing batch size checks here) ...
    # num_gpus = torch.cuda.device_count()
    # assert (config.loader.global_batch_size
    #         == (config.loader.batch_size
    #             * config.trainer.num_nodes
    #             * num_gpus
    #             * config.trainer.accumulate_grad_batches))
    # if config.loader.global_batch_size % (
    #         num_gpus * config.trainer.accumulate_grad_batches) != 0:
    #     raise ValueError(
    #         f'Train Batch Size {config.training.batch_size}'
    #         f'not divisible by {num_gpus} gpus with accumulation '
    #         f'{config.trainer.accumulate_grad_batches}.')
    # if config.loader.eval_global_batch_size % num_gpus != 0:
    #     raise ValueError(
    #         f'Eval Batch Size for {config.eval.batch_size} '
    #         f'not divisible by {num_gpus}.')



    # --- 1. LOAD DATASETS ---
    if skip_train:
        train_set = None
    else:
        # Pass streaming flag from config
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode='train',
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming  # <--- Ensure this is passed
        )

    if config.data.valid in ['text8', 'lm1b', 'ag_news']:
        validation_split = 'test'
    else:
        validation_split = 'validation'

    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming  # <--- Consistent streaming
        )

    # --- 2. CRITICAL: SHARD FOR DISTRIBUTED STREAMING ---
    # Since we can't use DistributedSampler, we must shard the dataset itself.
    if config.data.streaming and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        LOGGER.info(f"Sharding streaming dataset: Rank {rank}/{world_size}")

        if train_set is not None:
            # Each GPU gets 1/N of the data
            train_set = train_set.shard(num_shards=world_size, index=rank)

        if valid_set is not None:
            # Validation sharding (optional, prevents evaluating duplicate data)
            valid_set = valid_set.shard(num_shards=world_size, index=rank)

    # coupling = EmptyCoupling()
    coupling = UniformCoupling(max_len=config.model.length, vocab_size=tokenizer.vocab_size,
                               pad_token=tokenizer.pad_token_id, min_len=1, gap_token=3)

    seq_align_fn = functools.partial(opt_align_xs_to_zs, gap_token=3)

    collate_fn = functools.partial(collate_edit_batch,
                                   seq_align_fn=seq_align_fn,
                                   coupling=coupling,
                                   pad_token=tokenizer.pad_token_id,
                                   vocab_size=tokenizer.vocab_size,
                                   gap_token_id=3
                                   )

    if skip_train:
        train_loader = None
    else:
        # FOR STREAMING:
        # 1. shuffle must be False (we handled shuffling via buffer in get_dataset)
        # 2. sampler must be None (DataLoader handles fetching)
        should_shuffle = (not config.data.streaming)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=should_shuffle,  # Must be False for IterableDataset
            persistent_workers=True,
            collate_fn=collate_fn
        )
        train_loader.tokenizer = tokenizer

    if skip_valid:
        valid_loader = None
    else:
        # Logic for validation seed (only applies if NOT streaming)
        if valid_seed is None or config.data.streaming:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
            collate_fn=collate_fn
        )
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# def get_dataloaders(config, tokenizer, skip_train=False,
#                     skip_valid=False, valid_seed=None):
#     num_gpus = torch.cuda.device_count()
#     assert (config.loader.global_batch_size
#             == (config.loader.batch_size
#                 * config.trainer.num_nodes
#                 * num_gpus
#                 * config.trainer.accumulate_grad_batches))
#     if config.loader.global_batch_size % (
#             num_gpus * config.trainer.accumulate_grad_batches) != 0:
#         raise ValueError(
#             f'Train Batch Size {config.training.batch_size}'
#             f'not divisible by {num_gpus} gpus with accumulation '
#             f'{config.trainer.accumulate_grad_batches}.')
#     if config.loader.eval_global_batch_size % num_gpus != 0:
#         raise ValueError(
#             f'Eval Batch Size for {config.eval.batch_size} '
#             f'not divisible by {num_gpus}.')
#     if skip_train:
#         train_set = None
#     else:
#         train_set = get_dataset(
#             config.data.train,
#             tokenizer,
#             mode='train',
#             wrap=config.data.wrap,
#             cache_dir=config.data.cache_dir,
#             block_size=config.model.length,
#             streaming=config.data.streaming)
#
#     if config.data.valid in ['text8', 'lm1b', 'ag_news']:
#         validation_split = 'test'
#     else:
#         validation_split = 'validation'
#     if skip_valid:
#         valid_set = None
#     else:
#         valid_set = get_dataset(
#             config.data.valid,
#             tokenizer,
#             wrap=config.data.wrap,
#             mode=validation_split,
#             cache_dir=config.data.cache_dir,
#             block_size=config.model.length,
#             streaming=config.data.streaming)
#
#     coupling = EmptyCoupling()
#
#     seq_align_fn = functools.partial(opt_align_xs_to_zs, gap_token=3)  # todo fixed for now
#
#     collate_fn = functools.partial(collate_edit_batch,
#                                    seq_align_fn=seq_align_fn,
#                                    coupling=coupling,
#                                    pad_token=tokenizer.pad_token_id,
#                                    vocab_size=tokenizer.vocab_size,
#                                    gap_token_id=3
#                                    )
#
#     if skip_train:
#         train_loader = None
#     else:
#         train_loader = torch.utils.data.DataLoader(
#             train_set,
#             batch_size=config.loader.batch_size,
#             num_workers=config.loader.num_workers,
#             pin_memory=config.loader.pin_memory,
#             shuffle=not config.data.streaming,
#             persistent_workers=True,
#             collate_fn=collate_fn
#         )
#         train_loader.tokenizer = tokenizer
#     if skip_valid:
#         valid_loader = None
#     else:
#         if valid_seed is None:
#             shuffle_valid = False
#             generator = None
#         else:
#             shuffle_valid = True
#             generator = torch.Generator().manual_seed(valid_seed)
#         valid_loader = torch.utils.data.DataLoader(
#             valid_set,
#             batch_size=config.loader.eval_batch_size,
#             num_workers=config.loader.num_workers,
#             pin_memory=config.loader.pin_memory,
#             shuffle=shuffle_valid,
#             generator=generator,
#             collate_fn=collate_fn
#         )
#         # Will be used in generative perplexity calculation
#         valid_loader.tokenizer = tokenizer
#
#     return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
