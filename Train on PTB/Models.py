import torch
import torch.nn as nn
import logging
import math

class SaSLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(SaSLayer, self).__init__()
        self.m = nn.Parameter(torch.Tensor(inFeatures, outFeatures))
        self.pi = nn.Parameter(torch.Tensor(inFeatures, outFeatures))
        self.chi = nn.Parameter(torch.Tensor(inFeatures, outFeatures))
        # self.pi = torch.zeros(size=(inFeatures, outFeatures), device="cuda")
        # self.chi = torch.zeros(size=(inFeatures, outFeatures), device="cuda")
        self.inFeature = inFeatures
        self.resetParameters()
    
    def resetParameters(self):
        nn.init.uniform_(self.m, -1 / math.sqrt(self.inFeature), 1 / math.sqrt(self.inFeature))
        # nn.init.uniform_(self.pi, 0, 0.1)
        # nn.init.uniform_(self.chi, 0, 0.1)
        nn.init.zeros_(self.pi)
        nn.init.zeros_(self.chi)

    def forward(self, inputs, epsilon=None):
        mu = (1 - self.pi) * self.m
        rho = (1 - self.pi) * (self.chi + self.m**2)

        G = torch.matmul(inputs, mu)
        DeltaSquare = torch.matmul(inputs**2, rho - (mu**2))
        # torch.clamp_(DeltaSquare, 0)

        if epsilon is None:
            epsilon = torch.normal(0, 1, size=DeltaSquare.size(), device=inputs.device)
        z = G + epsilon * torch.sqrt(DeltaSquare + 1e-10)

        # z = torch.matmul(inputs, self.m)

        return z

    def clamp(self):
        self.pi.data.clamp_(min=0, max=1)
        self.chi.data.clamp_(min=0)

class RNNSaSLayer(nn.Module):
    def __init__(self, inputSizes, hiddenSize, nonlinearity, batchFirst):
        super(RNNSaSLayer, self).__init__()

        self.inputLayer = SaSLayer(inputSizes, hiddenSize)
        self.hiddenLayer = SaSLayer(hiddenSize, hiddenSize)
        # self.inputLayer = nn.Linear(inputSizes, hiddenSize)
        # self.hiddenLayer = nn.Linear(hiddenSize, hiddenSize)
        
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.hiddenSize = hiddenSize

    def forward(self, x, status=None):
        '''
        Parameters:
            x: shape is [sequence length, batch sizes, input sizes]
        '''
        sequenceLength, batchSizes, _ = x.size()
        device = next(self.parameters()).device

        x = self.inputLayer(x)
        h = torch.zeros((batchSizes, self.hiddenSize), device=device)
        outputs = []
        for s in range(sequenceLength):
            h = self.nonlinearity(self.hiddenLayer(h) + x[s])
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, h

    def clamp(self):
        self.inputLayer.clamp()
        self.hiddenLayer.clamp()

class RNNLayerForPC(nn.Module):
    def __init__(self, inputSizes, hiddenSize, nonlinearity, isSaS=False):
        super(RNNLayerForPC, self).__init__()

        if isSaS:
            self.inputLayer = SaSLayer(inputSizes, hiddenSize)
            self.hiddenLayer = SaSLayer(hiddenSize, hiddenSize)
        else:
            self.inputLayer = nn.Linear(inputSizes, hiddenSize, bias=False)
            self.hiddenLayer = nn.Linear(hiddenSize, hiddenSize, bias=False)
        
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.hiddenSize = hiddenSize
        self.isSaS = isSaS

    def forward(self, x, believes=None, epsilons=None, mode="inference"):
        '''
        Parameters:
            x: shape is [sequence length, batch sizes, input sizes]
        '''
        sequenceLength, batchSizes, _ = x.size()
        if epsilons is None:
            epsilonInput = None
            epsilonHidden = [None] * sequenceLength
        else:
            epsilonInput, epsilonHidden = epsilons
        
        if self.isSaS:
            x = self.inputLayer(x, epsilon=epsilonInput)
        else:
            x = self.inputLayer(x)
        if mode == "forward":
            prediction = []
            for s in range(sequenceLength):
                if s == 0:
                    h = x[s]
                else:
                    if self.isSaS:
                        h = self.hiddenLayer(self.nonlinearity(h), epsilon=epsilonHidden[s]) + x[s]
                    else:
                        h = self.hiddenLayer(self.nonlinearity(h)) + x[s]
                prediction.append(h.unsqueeze(0))

            prediction = torch.cat(prediction, dim=0)

            return prediction, self.nonlinearity(prediction)
        elif mode == "inference":
            prediction = []
            for s in range(sequenceLength):
                if s == 0:
                    h = x[s]
                else:
                    if self.isSaS:
                        h = self.hiddenLayer(self.nonlinearity(believes[s-1]), epsilon=epsilonHidden[s]) + x[s]
                    else:
                        h = self.hiddenLayer(self.nonlinearity(believes[s-1])) + x[s]
                prediction.append(h.unsqueeze(0))

            prediction = torch.cat(prediction, dim=0)

            return prediction, self.nonlinearity(prediction)        


    def clamp(self):
        if self.isSaS:
            self.inputLayer.clamp()
            self.hiddenLayer.clamp()

class EmbeddingSaSLayer(nn.Module):
    def __init__(self, vocabularySize, embeddingSize):
        super(EmbeddingSaSLayer, self).__init__()

        self.weight = SaSLayer(vocabularySize, embeddingSize)
        mask = torch.eye(vocabularySize)
        self.register_buffer("mask", mask)

    def forward(self, inputs):
        outputs = self.weight(self.mask[inputs])

        return outputs
    
    def clamp(self):
        self.weight.clamp()

class BaseLanguageModel(nn.Module):
    def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False):
        super(BaseLanguageModel, self).__init__()

        self.batchFirst = batchFirst
        self.vocabularySize = vocabularySize
        self.embeddingSize = embeddingSize

        if preTrainEmbedding is None:
            self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        else:
            self.embedding = nn.Embedding.from_pretrained(preTrainEmbedding, freeze=fixedEmbedding)

        # self.embedding = EmbeddingSaSLayer(vocabularySize, embeddingSize)

class BaseRNNModel(BaseLanguageModel):
    """
        Base class for model architecture of traditional rnn and predictive coding.
    """

    def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False, hiddenSize=128, **kwargs):
        super(BaseRNNModel, self).__init__(vocabularySize, embeddingSize, preTrainEmbedding, fixedEmbedding, batchFirst)
        self.hiddenSize = hiddenSize

        self.dropout = nn.Dropout(0.1)

class VanillaRNN(BaseRNNModel):
    '''
        Traditional RNN architecture.
    '''

    def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False, hiddenSize=128, rnnsTypes = "rnn", isSaS=False, **kwargs):
        super(VanillaRNN, self).__init__(vocabularySize, embeddingSize, preTrainEmbedding, fixedEmbedding, batchFirst, hiddenSize)

        if isSaS:
            self.rnn = RNNSaSLayer(embeddingSize, hiddenSize, nonlinearity='Tanh', batchFirst=batchFirst)
            self.readout = SaSLayer(hiddenSize, vocabularySize)
        else:
            self.rnn = getattr(nn, rnnsTypes.upper())(embeddingSize, hiddenSize, batch_first=batchFirst)
            self.readout = nn.Linear(hiddenSize, vocabularySize)

        self.isSaS = isSaS

    def forward(self, x, states=None):
        # Embed word ids to vectors
        x = self.embedding(x)
        
        # Forward propagate rnn
        out, states = self.rnn(x, states)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(-1, out.size(2))
        
        # out = self.dropout(out)

        # Decode hidden states of all time steps
        out = self.readout(out)

        return out, states
    
    def clamp(self):
        if self.isSaS:
            self.rnn.clamp()
            self.readout.clamp()

class PCRNN(BaseRNNModel):
    '''
        Thw model is based on predicting coding mechanism.
    '''
    
    def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False, hiddenSize=128, rnnsTypes = "rnn", isSaS=False, T=20, eta=0.1, **kwargs):
        # With convience of inference, we use batchFirst=False
        super(PCRNN, self).__init__(vocabularySize, embeddingSize, preTrainEmbedding, fixedEmbedding, batchFirst, hiddenSize)
        # But store the user's setting (father class)

        self.T = T
        self.eta = eta
        self.rnnTypes = rnnsTypes
        self.isSaS = isSaS

        self.rnn = RNNLayerForPC(embeddingSize, hiddenSize, nonlinearity='Tanh', isSaS=isSaS)
        self.nonLinearity = self.rnn.nonlinearity
        if isSaS:
            self.readout = SaSLayer(hiddenSize, vocabularySize)
        else:
            self.readout = nn.Linear(hiddenSize, vocabularySize)
        

    def forward(self, *args, mode="predict"):
        # With convience of training program, we seperate the inference and prediction.
        if mode == "train":
            return self.inferenceLearning(*args)
        elif mode == "predict":
            return self.predict(*args)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def inferenceLearning(self, inputs, targets, lossFun):
        '''
            Inference learning.
            $x^t$: belief
            $\mu^t$: prediction
        '''
        
        # swap batch and sequence dimension
        if self.batchFirst == True:
            inputs = inputs.transpose(0, 1)
            targets = targets.transpose(0, 1)

        # forward propagate to generate pre-activation value as initial prediction value
        sequenceLength, batchSizes = inputs.size()
        device = inputs.device
        epsilonInput = torch.normal(0, 1, size=(sequenceLength, batchSizes, self.hiddenSize), device=inputs.device)
        epsilonHidden = torch.normal(0, 1, size=(sequenceLength, batchSizes, self.hiddenSize), device=inputs.device)
        epsilonOutput = torch.normal(0, 1, size=(sequenceLength * batchSizes, self.vocabularySize), device=inputs.device)

        # initialize belief
        x_i = inputs
        x_e = self.embedding(x_i)
        x_h, _ = self.rnn(x_e, mode="forward", epsilons=(epsilonInput, epsilonHidden))
        x_y = targets # special for last layers

        # backward propagate by time step
        x_i = x_i.detach().clone()
        x_e = x_e.detach().clone()
        x_h = x_h.detach().clone().requires_grad_()
        x_y = x_y.detach().clone()
        optimizerWithBelieves = torch.optim.Adam(iter([x_h]), lr=self.eta)
        lastFreeEnergy = 10
        for t in range(self.T):
            # prediction
            # embedding layer
            x_e = self.embedding(x_i)
            # epsilon_e = x_e - mu_e
                
            # recurrent hidden layer
            mu_h, outputs = self.rnn(x_e, x_h, mode="inference", epsilons=(epsilonInput, epsilonHidden))
            epsilon_h = x_h - mu_h

            # readout layer
            # outputs = self.dropout(outputs)
            if self.isSaS:
                mu_y = self.readout(outputs.reshape(sequenceLength * batchSizes, self.hiddenSize), epsilon=epsilonOutput)
            else:
                mu_y = self.readout(outputs.reshape(sequenceLength * batchSizes, self.hiddenSize))
            loss = lossFun(mu_y, x_y.reshape(-1))
            # epsilon_y = torch.autograd.grad(loss, mu_y, retain_graph=True)[0].detach().clone()

            # compute energy
            # F_e = torch.sum(torch.mean(epsilon_e**2, dim=(0, 1))) / 2
            F_h = torch.sum(torch.mean(epsilon_h**2, dim=(0, 1))) / 2
            # F = (F_e + F_h) + loss
            F = F_h + loss
            
            logging.debug("Iteration: {}, F_h: {}, F: {}, F: {}".format(t, F_h.item(), F.item(), loss.item()))

            # if lastFreeEnergy < F.item():
            #     break

            if t < (self.T - 1):
                optimizerWithBelieves.zero_grad()
                F.backward(inputs=[x_h])
                optimizerWithBelieves.step()

            lastFreeEnergy = F.item()

        logging.debug("{}, F_h: {}, loss: {}".format(t+1, F_h.item(), loss.item()))

        # update the model parameters
        for p in self.embedding.parameters():
            p.grad = torch.autograd.grad(F, p, retain_graph=True)[0]
        for p in self.rnn.parameters():
            p.grad = torch.autograd.grad(F, p, retain_graph=True)[0]
        for p in self.readout.parameters():
            p.grad = torch.autograd.grad(F, p, retain_graph=True)[0]

        # free memory
        with torch.no_grad():
            F = F.clone()

        return loss

    def setT(self, T):
        self.T = T

    def predict(self, inputs):
        '''
            prediction only
        '''
        
        outputs = self.embedding(inputs)
        if self.isSaS:
            _, outputs = self.rnn(outputs, mode="forward", epsilons=None)
            outputs = outputs.reshape(outputs.size(0)*outputs.size(1), outputs.size(2))
            outputs = self.readout(outputs, epsilon=None)
        else:
            _, outputs = self.rnn(outputs, mode="forward")
            outputs = outputs.reshape(outputs.size(0)*outputs.size(1), outputs.size(2))
            outputs = self.readout(outputs)

        return outputs
    
    def clamp(self):
        if self.isSaS:
            self.rnn.clamp()
            self.readout.clamp()

class CBOW(nn.Module):
    def __init__(self, windowSize, vocabularySize, embeddingSize):
        super(CBOW, self).__init__()

        self.windowSize = windowSize
        self.embedding = nn.Embedding(vocabularySize, embeddingSize)
        self.linear = nn.Linear(embeddingSize, vocabularySize)

    def forward(self, inputs):
        '''
            Inputs: [batchSizes, windowSize]
            HiddenBefore: [batchSizes, windowSize, embeddingSize]
            HiddenAfter: [batchSizes, embeddingSize]
            Outputs: [batchSize, vocabularySize]
        '''

        outputs = self.embedding(inputs)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.linear(outputs)

        return outputs

# class Transformer(BaseLanguageModel):
#     def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False, hiddenSize=128, numHead=1, **kwargs):
#         super(Transformer, self).__init__(vocabularySize, embeddingSize, preTrainEmbedding, fixedEmbedding, batchFirst)

#         # self.dropout = nn.Dropout(0.2)
        
#         encoderLayer = nn.TransformerEncoderLayer(embeddingSize, numHead, hiddenSize, dropout=0, batch_first=batchFirst)
#         self.encoder = nn.TransformerEncoder(encoderLayer, num_layers=1)

#         self.decoder = nn.Linear(embeddingSize, vocabularySize)

#     def forward(self, inputs):
#         outputs = self.embedding(inputs)
#         outputs = self.encoder(outputs)
#         # Reshape output to (batch_size*sequence_length, hidden_size)
#         outputs = outputs.reshape(-1, outputs.size(2))
#         outputs = self.decoder(outputs)

#         return outputs

class Transformer(BaseLanguageModel):
    def __init__(self, vocabularySize, embeddingSize, preTrainEmbedding=None, fixedEmbedding=False, batchFirst=False, hiddenSize=128, numHead=1, **kwargs):
        super(Transformer, self).__init__(vocabularySize, embeddingSize, preTrainEmbedding, fixedEmbedding, batchFirst)

        # Attention
        self.WQ = nn.Parameter(torch.Tensor(embeddingSize, embeddingSize))
        self.WK = nn.Parameter(torch.Tensor(embeddingSize, embeddingSize))
        self.WV = nn.Parameter(torch.Tensor(embeddingSize, embeddingSize))
        self.softmax = nn.Softmax()
        self.outProjection = nn.Linear(embeddingSize, embeddingSize)
        # Feedforward
        self.linear1 = nn.Linear(embeddingSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, embeddingSize)
        self.activation = nn.ReLU()
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)

        self.decoder = nn.Linear(embeddingSize, vocabularySize)

        self.embeddingSize = embeddingSize

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.WQ)
        torch.nn.init.xavier_uniform_(self.WK)
        torch.nn.init.xavier_uniform_(self.WV)

    def forward(self, inputs):
        sequenceLength, batchSizes = inputs.size()

        embeddings = self.embedding(inputs)

        # self attention
        Q = torch.nn.functional.linear(embeddings, self.WQ)
        K = torch.nn.functional.linear(embeddings, self.WK)
        V = torch.nn.functional.linear(embeddings, self.WV)
        Q = Q.contiguous().transpose(0, 1)
        K = K.contiguous().transpose(0, 1)
        V = V.contiguous().transpose(0, 1)
        z = (torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(self.embeddingSize)).softmax(dim=-1)
        z = torch.bmm(z, V)
        z = z.transpose(0, 1).contiguous().view(sequenceLength * batchSizes, self.embeddingSize)
        z = self.outProjection(z)
        z = z.view(sequenceLength, batchSizes, self.embeddingSize)

        # linear
        outputs = self.norm1(embeddings + z)
        outputs = self.norm2(outputs + self.linear2(self.activation(self.linear1(outputs))))
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        outputs = outputs.reshape(-1, outputs.size(2))
        outputs = self.decoder(outputs)

        return outputs