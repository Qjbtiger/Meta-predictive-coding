import torch
import argparse
from Corpus import convert2ids, getPennTreebankCorpus, getIMDBCorpus, refreshDataset, WordWindowDataset
import numpy as np
import time
from Models import CBOW
from Vocabulary import createVocab
from torch.utils.data import DataLoader

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train word embeddings by CBOW models")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["cuda:0", "mps", "cpu", "auto"], help="Device to use (cuda, mps, cpu or auto)")
    parser.add_argument("--dataset", "--ds", type=str, default="ptb", choices=["ptb", "imdb"], help="Dataset to use (ptb or imdb)")
    parser.add_argument("--batch_size", "--bs", type=int, default=128, help="Batch size of training set")
    parser.add_argument("--window_size", "--ws", type=int, default=2, help="Window size of each batch")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_size", "--es", type=int, default=300, help="Embedding size")
    parser.add_argument("--min_frequency", "--mf", type=int, default=5, help="the minimum frequence of word stay in corpus")
    args = parser.parse_args()

    # choice device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print('Current device: {}'.format(device))

    # parameter setting
    batchSizes = args.batch_size
    windowSize = args.window_size
    learningRate = args.learning_rate
    embeddingSize = args.embedding_size
    numEpoch = args.epochs
    minFrequence = args.min_frequency
    reportInterval = 200

    # load dataset
    if args.dataset == "ptb":
        trainDataset, testDataset = getPennTreebankCorpus()
    elif args.dataset == "imdb":
        trainDataset, testDataset = getIMDBCorpus()
        trainDataset.specials.append("<unk>")
        testDataset.specials.append("<unk>")
    else:
        raise ValueError("Invalid dataset")

    # create vocabulary
    vocabulary = createVocab([trainDataset, testDataset], minFrequence=minFrequence)
    trainDataset = refreshDataset(trainDataset, vocabulary, mode='<unk>')
    testDataset = refreshDataset(testDataset, vocabulary, mode='<unk>')
    print("Initial vocabulary size: {}".format(len(vocabulary)))
    
    # Convert dataset to ids
    trainDataset = convert2ids(trainDataset, vocabulary)
    testDataset = convert2ids(testDataset, vocabulary)

    # build dataset and dataloader
    trainDataset = WordWindowDataset(trainDataset, windowSize)
    testDataset = WordWindowDataset(testDataset, windowSize)
    trainDataloader = DataLoader(trainDataset, batch_size=batchSizes, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=batchSizes, shuffle=False)

    # model
    model = CBOW(windowSize, len(vocabulary), embeddingSize=embeddingSize).to(device)

    # Criterion and optimizer setting
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # result collectting
    numTrainBatches = len(trainDataloader)
    numTestBatches = len(testDataloader)
    numTrainDataset = len(trainDataset)
    numTestDataset = len(testDataset)
    trainLosses = []
    trainPreplexities = []
    testLosses = []
    testPreplexities = []

    # Train
    for t in range(numEpoch):
        start = time.time()
        lossSum = 0.0
        lastReportLossSum = 0.0
        model.train()
        for i, batch in enumerate(trainDataloader):
            inputs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lossSum += loss.item()

            if (i+1) % reportInterval == 0:
                currentLoss = (lossSum - lastReportLossSum) / reportInterval
                print("Epoch: {}, Batches: [{}/{}], Loss: {:.4f}, Perplexity: {:.4f}, Time: {:4.2f}\r".format(t+1, i+1, numTrainBatches, currentLoss, np.exp(currentLoss), time.time() - start), end='')
                lastReportLossSum = lossSum

        lossSum /= numTrainBatches
        print("Epoch: [{}/{}], Loss: {:.4f}, Perplexity: {:.4f}".format(t+1, numEpoch, lossSum, np.exp(lossSum)))
        trainLosses.append(lossSum)
        trainPreplexities.append(np.exp(lossSum))

        # Test
        lossSum = 0.0
        correct = 0
        model.eval()
        for i, batch in enumerate(testDataloader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            lossSum += loss.item()

            correct += torch.sum(torch.argmax(outputs, dim=1) == targets)
        
        lossSum /= numTestBatches
        print("[Test] Loss: {:.4f}, Perplexity: {:.4f}, Correct: {:.2%}".format(lossSum, np.exp(lossSum), correct / numTestDataset))
        testLosses.append(lossSum)
        testPreplexities.append(np.exp(lossSum))
    
    torch.save((vocabulary, list(model.embedding.parameters())[0].detach().cpu(), embeddingSize), "./Data/vocabulary/Word2Vec/{}-mf{}-w{}-d{}.pt".format(args.dataset, minFrequence, len(vocabulary), embeddingSize))
    
