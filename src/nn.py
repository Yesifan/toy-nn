from numpy import random, dot, array
import scipy.special

class neuralNetwork:
  # 默认激活函数
  activation = lambda self, x: scipy.special.expit(x)
  # 默认随机权重算法
  calculate_weight = lambda self, x, y: (random.rand(x, y) - 0.5)

  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.1) -> None:
    self.nodes_input = input_nodes
    self.nodes_hidden = hidden_nodes
    self.nodes_output = output_nodes
    self.lr = learning_rate

    self.weight_matrix = []

    for index, layer in enumerate(self.nodes_hidden):
      input = self.nodes_input if index == 0 else self.nodes_hidden[index-1]
      weight = self.calculate_weight(layer, input)
      self.weight_matrix.append(weight)
      
    layer_last = self.nodes_hidden[len(self.nodes_hidden)-1]
    weight_o = self.calculate_weight(self.nodes_output, layer_last)
    self.weight_matrix.append(weight_o)

  def train(self, input_list, target_list):
    target = array(target_list)
    output = self.query(input_list)
    error_o = target - output
    return error_o

  # 前向馈送信号
  def query(self, input_list):
    input = array(input_list)
    for weight in self.weight_matrix:
      input_w = dot(weight, input)
      input = self.activation(input_w)
    return input