import numpy as np

#I've tried to use logical names for my variables as much as possible, so I looked it up on the internet, to see some common names for
# specific parts.

#There is no theoretical explanation of the algorithm, because it is hard to explain it in this way and what is more, it would be space consuming.
#Theoretical explanation shall be given during presentation of the project.

class NeuralNetwork:
    def __init__(self,numberOfInputNodes,numberOfHiddenNodes,numberOfOutputNodes,learningRate):
        self.numberOfInputNodes = numberOfInputNodes
        self.numberOfHiddenNodes = numberOfHiddenNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.learningRate = learningRate

        #0.0 = center point of distribution interval
        #self.numberOfInputNodes ** (-.5) = standard deviation, defines borders of distribution interval
        #third parameter represent shape of matrix
        self.weightsInputToHidden = np.random.normal(0.0, self.numberOfInputNodes ** (-.5),(self.numberOfInputNodes, self.numberOfHiddenNodes))
        self.weightsHiddenToOutput = np.random.normal(0.0, self.numberOfHiddenNodes ** (-.5),(self.numberOfHiddenNodes, self.numberOfOutputNodes))

        #np.zeros returns np.array filled with t zeros, where t is passed as parameter and represent length of np.array
        self.inputs = np.zeros(self.numberOfInputNodes)
        self.hidden = np.zeros(self.numberOfHiddenNodes)
        self.outputs = np.zeros(self.numberOfOutputNodes)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forwardPass(self,appliedInputs):

        #Firstly, we fill our input layer nodes with passed parameter inputs
        #shape of appliedInputs must correspond to the shape of input layer
        for i in range(self.numberOfInputNodes):
            self.inputs[i]=appliedInputs[i]

        #for every hidden node in our hidden layer, we calculate partialSum as input*weight for every edge between input layer and our hidden node
        #At the end, we put partialSum through activation function (in our case, it is sigmoid), in order to get value between [0,1]
        for i in range(self.numberOfHiddenNodes):
            partialSum=0.0
            for j in range(self.numberOfInputNodes):
                partialSum+=self.inputs[j]*self.weightsInputToHidden[j,i] #[j,i] mimic transpose matrix in matrix notation
            self.hidden[i]=self.sigmoid(partialSum)

        for i in range(self.numberOfOutputNodes):
            partialSum=0.0
            for j in range(self.numberOfHiddenNodes):
                partialSum+=self.hidden[j]*self.weightsHiddenToOutput[j,i]
            self.outputs[i]=self.sigmoid(partialSum)

    def backpropagation(self,target):
        #Our imaginary error set for hidden layer
        deltaHidden = np.zeros(self.numberOfHiddenNodes)
        #our imaginary error set for output layer
        deltaOutputs = np.zeros(self.numberOfOutputNodes)

        #SquareError is multiplied by 2 because we defined our cost function as (target - prediction)**2
        #Bear in mind, that we use gradient optimization algorithm, so we need to use derivative of cost function
        #which turns out to be just two times the same value without power of two.
        #Secondly, we could also delete this multiplication by two because, in theory, we could represent our cost function as 1/2*cost

        #Same thing, because of gradient optimization algorithm, we need derivate of our sigmoid function
        # and that is effectively = sig(x)*(1-sig(x))
        for i in range(self.numberOfOutputNodes):
            squareError = 2*(target[i]-self.outputs[i])
            deltaOutputs[i] = self.outputs[i]*(1.0-self.outputs[i])*squareError

        #Now, we calculate partialErrors in the same way as we did it in forwardPass, just in another direction.
        for i in range(self.numberOfHiddenNodes):
            partialError = 0.0
            for j in range(self.numberOfOutputNodes):
                partialError+=self.weightsHiddenToOutput[i,j]*deltaOutputs[j]
            deltaHidden[i]=self.hidden[i]*(1.0-self.hidden[i])*partialError

        #Again, [j,i] represent transpose matrix in matrix notation
        for i in range(self.numberOfOutputNodes):
            for j in range(self.numberOfHiddenNodes):
                self.weightsHiddenToOutput[j,i]+=self.learningRate*deltaOutputs[i]*self.hidden[j]

        for i in range(self.numberOfHiddenNodes):
            for j in range(self.numberOfInputNodes):
                self.weightsInputToHidden[j,i]+=self.learningRate*deltaHidden[i]*self.inputs[j]

    def train(self,features,targets,iterations,error=False):
        #Zip give us ability to easily define two matching for loops
        for i in range(iterations):
            for x, y in zip(features, targets):
                self.forwardPass(x)
                self.backpropagation(y)
                mistake = np.array(y)-np.array(self.outputs)
                mistake = np.power(mistake,2)
                if error:
                    print(mistake)
                    print("Current mistake: ", np.sum(mistake))

    def run(self,features):
        for x in features:
            print("Input: ", x)
            self.forwardPass(x)
            print("Prediction: ",self.outputs)
            print()


#P.S. I know that this code is bad, slow and can't be use for production purposes. But I really tried to make it simple as much as possible, and
# put it in simple way. Also, I think that this type of implementation provides enough evidence that someone understand core concepts of neural networks.
