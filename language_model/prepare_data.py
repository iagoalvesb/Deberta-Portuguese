# coding: utf-8
from DeBERTa import deberta

import sys
import argparse
from tqdm import tqdm
import sentencepiece as spm

import sys


def tokenize_data(input, output=None, max_seq_length=512):  
  vocab_path, vocab_type = deberta.load_vocab(vocab_path='tokenizer/spm.model', vocab_type='spm')
  tokenizer = deberta.tokenizers[vocab_type](vocab_path)
  seq_length = max_seq_length - 2     # 2 tokens especiais
  
  remaining_tokens = []
  with open(input , encoding = 'utf-8') as fs:
      with open(output, 'w', encoding = 'utf-8') as wfs:
          for l in tqdm(fs, ncols=80, desc='Loading'):
              if len(l) > 0:
                  tokens = tokenizer.tokenize(l)
              else:
                  tokens = []

              remaining_tokens.extend(tokens)

              while len(remaining_tokens) >= seq_length:
                  wfs.write(' '.join(remaining_tokens[:seq_length]) + '\n')
                  remaining_tokens = remaining_tokens[seq_length:]

  #print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input data path')
parser.add_argument('-o', '--output', default=None, help='The output data path')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
args = parser.parse_args()
tokenize_data(args.input, args.output, args.max_seq_length)
