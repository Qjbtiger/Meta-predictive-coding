import random
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import PennTreebank, IMDB
import torch
from torchtext.transforms import Sequential, AddToken
from torchtext.data.utils import get_tokenizer

class WordWindowDataset(torch.utils.data.Dataset):
    def __init__(self, rawDataset, windowSize):
        super(WordWindowDataset, self).__init__()
        self.windowSize = windowSize
        rawdata =  torch.cat([torch.tensor(text, dtype=torch.int64) for text in rawDataset]).contiguous()
        self.numDatas = rawdata.size(0) - self.windowSize * 2
        self.datas = torch.stack([torch.cat([rawdata[i:i+self.windowSize], rawdata[i+self.windowSize+1:i+self.windowSize+self.windowSize+1]]) for i in range(self.numDatas)])
        self.labels = torch.stack([rawdata[i] for i in range(self.numDatas)])

    def __getitem__(self, i):
        return self.datas[i], self.labels[i]

    def __len__(self):
        return self.numDatas

# Self-define class to avoid problem 
class _MapStyleDataset(torch.utils.data.Dataset):
    def __init__(self, iter_data) -> None:
        # TODO Avoid list issue #1296
        self._data = list(iter_data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

def getPennTreebankCorpus():
    '''
        Get PennTreebank dataset.
    '''
    tokenizer = get_tokenizer("basic_english", language='en')
    transform = Sequential(
        # AddToken(token="<bos>", begin=True),
        AddToken(token="<eos>", begin=False)
    )

    def textTransform(textDatapipe):
        # split and add sentense ending symbol <eos>
        textDatapipe = textDatapipe.map(tokenizer).map(transform)
    
        # return to_map_style_dataset(textDatapipe)
        return _MapStyleDataset(textDatapipe)
    
    trainDatapipe, testDatapipe = PennTreebank(root='./Data/', split=('train', 'test'))
    
    trainDataset = textTransform(trainDatapipe)
    testDataset = textTransform(testDatapipe)
    
    trainDataset.specials = ['<unk>', '<bos>', '<eos>']
    testDataset.specials = ['<unk>', '<bos>', '<eos>']

    return trainDataset, testDataset

def getIMDBCorpus():
    '''
        Get IMDB dataset.
    '''

    tokenizer = get_tokenizer("basic_english", language='en')
    transform = AddToken(token="<eos>", begin=False)
    
    def tokenizerIter(textDatapipe):
        for _, text in textDatapipe:
            yield transform(tokenizer(text)) # remove label

    def textTransform(textDatapipe):
        # textDataset = to_map_style_dataset(tokenizerIter(textDatapipe)) 
        textDataset = _MapStyleDataset(tokenizerIter(textDatapipe)) 
        
        return textDataset
    
    trainDatapipe, testDatapipe = IMDB(root='./Data/', split=('train', 'test'))

    trainDataset = textTransform(trainDatapipe)
    testDataset = textTransform(testDatapipe)

    trainDataset.specials = ['<eos>', ]
    testDataset.specials = ['<eos>', ]

    return trainDataset, testDataset

def refreshDataset(dataset, vocabulary, mode='<unk>'):
    '''
        Replace  or Remove tokens which are not in vocabulary.

    Args:
        dataset: Dataset to be refreshed.
        vocabulary: Vocabulary to be used.
        mode: '<unk>' or 'remove'
            '<unk>': Replace tokens which are not in vocabulary with <unk>
            'remove': Remove tokens which are not in vocabulary

    Returns:
        new dataset
    '''

    if not (mode == '<unk>' or mode == 'remove'):
        raise ValueError('mode must be \'<unk>\' or \'remove\'')
    
    wordList = set(vocabulary.get_itos())

    if mode == '<unk>':
        for text in dataset:
            for i, token in enumerate(text):
                if token not in wordList:
                    text[i] = '<unk>'
    elif mode == 'remove':
        for t, text in enumerate(dataset):
            newText = []
            for token in enumerate(text):
                if token in wordList:
                    newText.append(token)
            dataset[t] = newText
    
    return dataset

def batchify(dataset, batchSizes, shuffle=False):
    '''
        Change dataset into regular lines (sequenceLength * batchSiz)
    '''

    if shuffle:
        random.shuffle(dataset._data)

    # change to one line include all sentences
    dataset = torch.cat([torch.tensor(text, dtype=torch.int64) for text in dataset])

    # Work out how cleanly we can divide the dataset into bsz parts.
    numBatch = dataset.size(0) // batchSizes
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    dataset = dataset.narrow(0, 0, numBatch * batchSizes)
    # Evenly divide the data across the bsz batches.
    dataset = dataset.view(batchSizes, -1).t().contiguous()

    return dataset

def convert2ids(dataset, vocabulary):
    '''
        Convert dataset to ids.
    '''
    
    for text in dataset:
        for i, token in enumerate(text):
            text[i] = vocabulary[token]
    
    return dataset