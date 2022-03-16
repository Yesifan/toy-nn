from numpy import random, dot, array
import scipy.special

class neuralNetwork:
  # 默认激活函数
  activation = lambda self, x: scipy.special.expit(x)
  # 默认随机权重算法
  calculate_weight = lambda self, x, y: (random.rand(x, y) - 0.5)

  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.001) -> None:
    self.nodes_input = input_nodes
    self.nodes_hidden = hidden_nodes
    self.nodes_output = output_nodes
    self.lr = learning_rate

    self.weight_list = []

    for index, layer in enumerate(self.nodes_hidden):
      input = self.nodes_input if index == 0 else self.nodes_hidden[index-1]
      weight = self.calculate_weight(layer, input)
      self.weight_list.append(weight)
      
    layer_last = self.nodes_hidden[len(self.nodes_hidden)-1]
    weight_o = self.calculate_weight(self.nodes_output, layer_last)
    self.weight_list.append(weight_o)

  def run(self, input_list):
    input = array(input_list, ndmin=2).T
    output_list = self.query(input)
    return output_list[len(output_list)-1]

  def train(self, input_list, target):
    input = array(input_list, ndmin=2).T
    target_t = array(target, ndmin=2).T

    output_list = self.query(input)

    output = output_list[len(output_list)-1]
    error = target_t - output
  
    modify_list = self.modify(error, input, output_list)

    modify_weigth_list = list(map(lambda x,y: x + y, self.weight_list, modify_list))
    self.weight_list = modify_weigth_list

  # 前向馈送信号
  def query(self, input):
    current = input
    output_list = []
    for weight in self.weight_list:
      input_w = dot(weight, current)
      current = self.activation(input_w)
      output_list.append(current)
    return output_list
  
  # 反向误差传播
  def modify(self, error, input, output_list):
    error_list = []
    modify_list = []
    current_error = error

    for weight in reversed(self.weight_list):
      error_list.insert(0, current_error)
      current_error = dot(weight.T, current_error)

    for index, current in enumerate(output_list):
      error = error_list[index]
      output_prev = input.T if index == 0 else output_list[index-1].T
      modify = self.lr * dot(error * current * (1-current), output_prev)
      modify_list.append(modify)
    return modify_list