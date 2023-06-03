import os
from typing import List, Text, Tuple

import joblib
from sinling import SinhalaTokenizer
from sinling.config import RESOURCE_PATH
from sinling.sinhala.stemmer import SinhalaStemmer

tokenizer = SinhalaTokenizer()

__all__ = ['Tagger']


class Tagger:
    def predict(self, tokens: List[Text]) -> List[Tuple[Text, Text]]:
        raise NotImplementedError


__all__ = ['POSTagger']


class POSTagger(Tagger):
    def __init__(self):
        self._model = joblib.load(os.path.join(RESOURCE_PATH, 'pos-tagger-crf-sinling.joblib'))
        self._stemmer = SinhalaStemmer()

    def predict(self, x: List[List[Text]]) -> List[List[Tuple[Text, Text]]]:
        features = [[self._word2features(ts, i) for i in range(len(ts))] for ts in x]
        pos_tags = self._model.predict(features)
        return [list(zip(x[ix], pos_tags[ix])) for ix in range(len(x))]

    def _word2features(self, sent, i):
        word = sent[i]
        stem, suff = self._stemmer.stem(word)
        features = {
            'bias': 1.0,
            word: True,
            f'STEM': stem,
            f'SUFF': suff,
            'len(word)': len(word),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word_prev = sent[i - 1]
            features.update({
                f'-1:word': word_prev,
                '-1:word.isdigit()': word_prev.isdigit(),
            })
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            word_next = sent[i + 1]
            features.update({
                f'+1:word': word_next,
                '+1:word.isdigit()': word_next.isdigit(),
            })
        else:
            features['EOS'] = True
        return features


# document = 'ගරසරප චිත්‍රපට පහල තියෙ සම්බන්ධක එක  බාගත කරග ඇතැම් වෛරස් රෝග වලට වැක්සීන හෙවත් එන්නත් ද වෛරස් නාශක ඖෂධ ද ' \
#            'තිබුනද සියලූ‍ වෛරස් සම්බන්ධයෙන් ඒ න්‍යාය වැඩ කරන්නේ නැත. වසූරිය වෛරසය මිනිසා විසින් මිහිමතින් තුරන් කර ' \
#            'තිබේ.'

# tokenized_sentences = [tokenizer.tokenize(f'{ss}.') for ss in tokenizer.split_sentences(document)]
# print(tokenized_sentences)
tagger = POSTagger()


# pos_tags = tagger.predict(tokenized_sentences)
#
# for sent in pos_tags:
#     print('------------------')
#     print(sent)

def get_pos_tags(text: str) -> List[Tuple[str, str]]:
    tokenized_sentences = [tokenizer.tokenize(f'{ss}.') for ss in tokenizer.split_sentences(text)]
    pos_tags = tagger.predict(tokenized_sentences)
    return pos_tags
