import torch
import easydict
import time
from Corpus import getIMDBCorpus, getPennTreebankCorpus, refreshDataset, convert2ids, batchify
from Vocabulary import createVocab, loadVectors, overlapOfVocabs
from Models import VanillaRNN, PCRNN, Transformer
import math
import logging
import os
import random

class Record(easydict.EasyDict):
    def __init__(self):
        super(Record, self).__init__()

    def add(self, newData: dict):
        for k, v in newData.items():
            if not k in self.keys():
                self[k] = []
            self[k].append(v)

def detachTensors(states):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if states == None:
        return None
    elif isinstance(states, torch.Tensor):
        return states.detach()
    else:
        return tuple(detachTensors(v) for v in states)

# torch.autograd.set_detect_anomaly(True)

def training(args):
    # choice device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.debug('Current device: {}'.format(device))

    filePath = f"./Data/preprocess/{args.dataset}-mf{args.minFrequence}.pt"
    if os.path.exists(filePath):
        rawTrainDataset, rawTestDataset, vocabulary = torch.load(filePath)
        vectors = None
    else:
        # Load dataset
        if args.dataset == "ptb":
            rawTrainDataset, rawTestDataset = getPennTreebankCorpus()
        elif args.dataset == "imdb":
            rawTrainDataset, rawTestDataset = getIMDBCorpus()
            rawTrainDataset.specials.append('<unk>')
            rawTestDataset.specials.append('<unk>')
        else:
            raise ValueError("Invalid dataset")

        # Create vocabulary and load pre-trained embeddings
        vocabulary = createVocab([rawTrainDataset, rawTestDataset], minFrequence=args.minFrequence)
        # if args.preTrained:
        #     if args.preTrained == 'glove':
        #         glove = GloVe(name='840B', dim=300, cache='./Data/vocabulary')
        #         vocabulary = overlapOfVocabs(vocabulary, glove, specials=rawTrainDataset.specials)
        #         print('Filtered vocabulary size: {}'.format(len(vocabulary)))

        #         vectors = loadVectors(vocabulary, glove, specials=rawTrainDataset.specials)
        #         vectors[0] = torch.mean(vectors[len(rawTrainDataset.specials)-1:], dim=0) # unkEmbedding
        #         vectors[0] = torch.randn(vectors.size(1)) # eosEmbedding
        #     else:
        #         # local word vector
        #         vocabulary, vectors, embeddingSize = torch.load('./Data/vocabulary/{}'.format(args.pre_trained)) # override parameters vocabulary embeddingSize
        #         args.pre_trained = "path"

        #     # reflash embedding size to pre-trained
        #     args.embeddingSize = vectors.size(1)
        # else:
        vectors = None
    
        # Covert dataset to ids
        rawTrainDataset = refreshDataset(rawTrainDataset, vocabulary, mode='<unk>')
        rawTestDataset = refreshDataset(rawTestDataset, vocabulary, mode='<unk>')
        rawTrainDataset = convert2ids(rawTrainDataset, vocabulary)
        rawTestDataset = convert2ids(rawTestDataset, vocabulary)

        torch.save((rawTrainDataset, rawTestDataset, vocabulary), filePath)

    logging.info('Vocabulary size: {}'.format(len(vocabulary)))

    # Set model
    args.isPredictiveCoding = (args.model in ["pc", "pcsas"])
    args.isSaS = (args.model in ["sas", "pcsas"])
    if args.model == "vanilla":
        if not args.rnnType in ["rnn", "lstm", "gru"]:
            raise ValueError("Invalid rnn type")
        
        Model = VanillaRNN
    elif args.model in ["pc", "sas", "pcsas"]:
        if not args.rnnType in ["rnn"]:
            raise ValueError("Invalid rnn type")
        
        Model = PCRNN if args.isPredictiveCoding else VanillaRNN
    elif args.model == "transformer":
        Model = Transformer
    else:
        raise ValueError("Invalid model")

    model = Model(
        vocabularySize=len(vocabulary), 
        embeddingSize=args.embeddingSize,
        preTrainEmbedding=vectors,
        fixedEmbedding=args.freezeEmbeddings,
        batchFirst=False,
        hiddenSize=args.hiddenSize,
        # special for rnn
        rnnsTypes=args.rnnType,
        isSaS=args.isSaS,
        # special for pc
        T=args.T,
        eta=args.eta,
        # special for transformer
        numHead=args.numHead
    ).to(device)

    # Criterier and optimizer setting
    criterier = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, cooldown=1, verbose=True)
    
    record = Record()

    if args.exportModelProgressing:
        torch.save(model, './Data/models/progressing/{}-{}-{}-{}-{}-ep{}.pt'.format(args.model, args.rnnType, args.dataset, "glove" if args.preTrained else "none", args.runId, 0))

    # Train
    minTestLoss = 10000
    for t in range(args.numEpoch):
        # Batchify dataset for each epoch (like random)
        trainDataset = batchify(rawTrainDataset, args.batchSizes, shuffle=True)
        testDataset = batchify(rawTestDataset, args.batchSizes)
        numTrainBatches = trainDataset.size(0) // args.sequenceLength
        numTestBatches = testDataset.size(0) // args.sequenceLength

        if not args.isPredictiveCoding == True:
            states = None

        if args.isPredictiveCoding == True:
            model.setT(args.T[t])

        start = time.time()
        totalTrainLoss = 0.0
        lastReportLossSum = 0.0
        model.train()
        for i, sequenceInitialIndex in enumerate(range(0, trainDataset.size(0) - 1, args.sequenceLength)):
            currentSequenceLength = min(args.sequenceLength, trainDataset.size(0) - 1 - sequenceInitialIndex)
            inputs = trainDataset[sequenceInitialIndex:sequenceInitialIndex + currentSequenceLength].to(device)
            targets = trainDataset[sequenceInitialIndex+1:sequenceInitialIndex + currentSequenceLength + 1].to(device)

            optimizer.zero_grad()

            if args.model in ["vanilla", "sas"]:
                states = detachTensors(states)
                outputs, states = model(inputs, states)

                loss = criterier(outputs, targets.reshape(-1))
                loss.backward()
            elif args.model == "transformer":
                outputs = model(inputs)

                loss = criterier(outputs, targets.reshape(-1))
                loss.backward()
            else:
                # pc, pcsas
                loss = model(inputs, targets, criterier, mode="train")

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            if args.isSaS:
                model.clamp()
            
            totalTrainLoss += loss.item()

            if (i+1) % args.reportInterval == 0:
                currentLoss = totalTrainLoss / args.reportInterval - lastReportLossSum
                logging.debug("Epoch: {}, Batches: [{}/{}], Loss: {:.4f}, Perplexity: {:.4f}, Time: {:4.2f}".format(t+1, i+1, numTrainBatches, currentLoss, math.exp(currentLoss), time.time() - start))
                lastReportLossSum += currentLoss

        totalTrainLoss /= numTrainBatches
        logging.info("Epoch: [{}/{}], Loss: {:.4f}, Perplexity: {:.4f}".format(t+1, args.numEpoch, totalTrainLoss, math.exp(totalTrainLoss)))

        # Test
        totalTestLoss = 0.0
        correct = 0
        model.eval()
        for i, sequenceInitialIndex in enumerate(range(0, testDataset.size(0) - 1, args.sequenceLength)):
            currentSequenceLength = min(args.sequenceLength, testDataset.size(0) - 1 - sequenceInitialIndex)
            inputs = testDataset[sequenceInitialIndex:sequenceInitialIndex + currentSequenceLength].to(device)
            targets = testDataset[sequenceInitialIndex+1:sequenceInitialIndex + currentSequenceLength + 1].to(device)
            
            if args.model in ["vanilla", "sas"]:
                outputs, _ = model(inputs, None)
            else:
                outputs = model(inputs)
            
            loss = criterier(outputs, targets.reshape(-1))
            totalTestLoss += loss.item()

            correct += torch.sum(torch.argmax(outputs, dim=1) == targets.reshape(-1)).item()
        
        totalTestLoss /= numTestBatches
        accurancy = correct / len(testDataset) / args.batchSizes
        logging.info("[Test] Loss: {:.4f}, Perplexity: {:.4f}, Accurancy: {:.2%}".format(totalTestLoss, math.exp(totalTestLoss), accurancy))
        record.add({
            "trainLosses": totalTrainLoss,
            "trainPerplexity": math.exp(totalTrainLoss),
            "testLosses": totalTestLoss,
            "testPerplexity": math.exp(totalTestLoss),
            "testAccurancy": accurancy
        })

        if totalTestLoss < minTestLoss:
            minTestLoss = totalTestLoss
            # Save models and embedding
            if args.exportModel:
                torch.save(model, './Data/models/{}-{}-{}-{}-e{}-h{}-{}.pt'.format(args.model, args.rnnType, args.dataset, "glove" if args.preTrained else "none", args.embeddingSize, args.hiddenSize, args.runId))
            if args.exportEmbedding:
                torch.save((vocabulary, list(model.embedding.parameters())[0].detach().clone().cpu(), args.embeddingSize), 
                "./Data/vocabulary/Train/{}-{}-{}-mf{}-w{}-d{}-{}.pt".format(args.model, args.rnnType, args.dataset, args.minFrequence, len(vocabulary), args.embeddingSize, args.runId))
        
        if args.exportModelProgressing:
            torch.save(model, './Data/models/progressing/{}-{}-{}-{}-{}-ep{}.pt'.format(args.model, args.rnnType, args.dataset, "glove" if args.preTrained else "none", args.runId, t+1))
        
        scheduler.step(totalTestLoss)

    if args.exportRecord:
        torch.save((record, args), 
                   './Data/record/{}-{}-{}-{}-e{}-h{}-{}.pt'
                   .format(args.model,
                           args.rnnType,
                           args.dataset, 
                           "glove" if args.preTrained else "none", 
                           args.embeddingSize, 
                           args.hiddenSize, 
                           args.runId))

    return record
