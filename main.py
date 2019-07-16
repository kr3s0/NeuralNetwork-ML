import NeuralNetwork as NN

#Random dataset with 3 inputs and 3 outputs


neuralna_mreza = NN.NeuralNetwork(3,12,3,0.5)
X = [[1,2,3],[4,1,2],[1,1,3],[5,2,4],[4,2,3],[1,5,3],[3,5,2]]
Y = [[0.9,0.8,0.7],[0.6,0.9,0.8],[0.9,0.9,0.7],[0.5,0.8,0.6],[0.6,0.8,0.7],[0.9,0.5,0.7],[0.7,0.5,0.8]]

"""
neuralna_mreza = NN.NeuralNetwork(3,6,1,0.5)
X = [[1,2,3],[4,1,2],[1,1,3],[5,2,4],[4,2,3],[1,5,3],[3,5,2]]
Y = [[1/2],[1/5],[1/3],[1/7],[1/5],[1/(-1)],[0]]
"""
"""
#Dataset for logic operations = AND,OR,XOR
neuralna_mreza = NN.NeuralNetwork(2,4,1,0.3)
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]
"""
neuralna_mreza.train(X,Y,5000,True)
neuralna_mreza.run(X)
#neuralna_mreza.run([[4,3,1],[5,3,3],[2,2,4],[4,5,1]])


