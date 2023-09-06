import torch
from torchtext.vocab import vocab
from collections import Counter

def createVocab(datasets, minFrequence=5):
    '''
        Create a vocabulary from the given datasets.
    '''

    counter = Counter() # Count the frequency of each token in the datasets
    # specials = set() # Special tokens that are not part of the vocabulary
    specials = []
    for dataset in datasets:
        for text in dataset:
            counter.update(text)
        # specials |= set(dataset.specials)
        for token in dataset.specials:
            if not token in specials:
                specials.append(token)

    return vocab(counter, min_freq=minFrequence, specials=specials)

def loadVectors(vocab, vectors, specials=None):
    '''
        Load pre-trained embeddings for the given vocabulary.
    '''

    newVectors = torch.Tensor(len(vocab), vectors.dim)
    for i, token in enumerate(vocab.get_itos()):
        if not token in specials:
            newVectors[i] = vectors.vectors[vectors.stoi[token.strip()]]
    
    return newVectors

def overlapOfVocabs(vocabulary, vectors, specials=None):
    """
        Return the overlap of two vocabularies.
    """

    vocab1 = set(vocabulary.get_itos())
    vocab2 = set(vectors.itos)
    overlap = vocab1 & vocab2
    overlap = dict([(token, 1) for token in overlap])

    return vocab(overlap, specials=specials)