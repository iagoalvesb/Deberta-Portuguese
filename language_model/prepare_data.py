# coding: utf-8
from DeBERTa import deberta

import sys
import argparse
from tqdm import tqdm
import sentencepiece as spm

import sys

class Tokenizer(object):
    def __init__(self, tokenizer_path) -> None:
        self.tokenizer_path = tokenizer_path
        self.sp = spm.SentencePieceProcessor(model_file=self.tokenizer_path)

    def tokenize(self, l) -> list:
        tokens = self.sp.encode_as_ids(l)
        return tokens

def tokenize_data(input, output=None, max_seq_length=512):  
#   vocab_path, vocab_type = deberta.load_vocab(vocab_path='tokenizer/spm.model', vocab_type='spm', pretrained_id='deberta-v3-base')
  remaining_tokens = []
  tokenizer = Tokenizer('deBeRta')

  with open('data_debertav3/train_wiki_brwac.raw', encoding = 'utf-8') as fs:
      with open('deberta_v3_pt_tokenized/train.txt', 'w', encoding = 'utf-8') as wfs:
          for l in tqdm(fs, ncols=80, desc='Loading'):
              if len(l) > 0:
                  tokens = tokenizer.tokenize(l)
              else:
                  tokens = []

              remaining_tokens.extend(tokens)

              while len(remaining_tokens) >= 510:
                  wfs.write(' '.join(remaining_tokens[:510]) + '\n')
                  remaining_tokens = remaining_tokens[510:]

  #print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input data path')
parser.add_argument('-o', '--output', default=None, help='The output data path')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
args = parser.parse_args()
tokenize_data(args.input, args.output, args.max_seq_length)
