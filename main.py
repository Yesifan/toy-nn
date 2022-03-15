from src.nn import neuralNetwork

input = 3
hidden = [3, 3, 2, 5]
output = 2

n = neuralNetwork(input, hidden, output)

output = n.train([0.3, 0.2, 0.4], [0.1,0.1])

print(output)