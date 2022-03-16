from src.nn import neuralNetwork

dataset = {
  'input':[0.3, 0.2, 0.4],
  'target': [0.1]
}

input = 3
hidden = [3, 3, 2]
output = 1

n = neuralNetwork(input, hidden, output)

for index in range(200000):
  n.train(dataset['input'], dataset['target'])

print(n.run([0.3, 0.2, 0.4]))
print(n.run([0.1, 0.8, 0.9]))