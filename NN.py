import numpy as NP

#This code does not work, because there is no part for updating weights between input,hidden and output layer.
#Apart from that, this piece of code has purpose to show how to implement Neural Network in as much as 50 lines of code using
# Matrix notation.



class NeuralNetwork:
    def __init__(self,numberOfInputNodes,numberOfHiddenNodes,numberOfOutputNodes,learningRate):
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfHiddenNodes = numberOfHiddenNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.learningRate = learningRate
        self.weightsInputToHidden = NP.random.normal(0.0,self.numberOfInputNodes**(-.5),(self.numberOfInputNodes,self.numberOfHiddenNodes))
        self.weightsHiddenToOutput = NP.random.normal(0.0,self.numberOfHiddenNodes**(-.5),(self.numberOfHiddenNodes,self.numberOfOutputNodes))

    def sigmoid(self,x):
        return 1/(1+NP.exp(-x))

    #vracam rezultate nakon hidden i output layera ali nakon sigmoid funkcije
    def forwardPass(self,features):
        hiddenResults = self.sigmoid(NP.dot(features,self.weightsInputToHidden))
        outputResults = self.sigmoid(NP.dot(hiddenResults,self.weightsHiddenToOutput))
        return hiddenResults, outputResults

    def backpropagation(self,hiddenResults,outputResults,features,targets):
        squareError = 2*(targets-outputResults)
        deltaOutputs = outputResults*(1.0-outputResults)*squareError
        hiddenError = deltaOutputs*self.weightsHiddenToOutput.T #da bi dobio errore h1 h2 h3 u obliku [h1 h2 h3]
        deltaHidden = hiddenResults*(1.0-hiddenResults)*hiddenError
        self.weightsHiddenToOutput = self.weightsHiddenToOutput+(self.learningRate*(deltaOutputs*hiddenResults))

    def train(self,features,targets):
        for i in range(5000):
            for X,Y in zip(features,targets):
                hiddenResults, outputResults = self.forwardPass(X)
                self.backpropagation(hiddenResults,outputResults,X,Y)

        print("Mreza je zavrsila:")
        h, o = self.forwardPass(features)
        print("Trebalo se dobiti:",targets)
        print("Rezultati su:",o)
