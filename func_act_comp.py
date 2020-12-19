import torch
import numpy as np

seed = int(input())
np.random.seed(seed)
torch.manual_seed(seed)

NUMBER_OF_EXPERIMENTS = 200

class SimpleNet(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.activation = activation
        self.fc1 = torch.nn.Linear(1, 1, bias=False)  # one neuron without bias
        self.fc1.weight.data.fill_(1.)  # init weight with 1
        self.fc2 = torch.nn.Linear(1, 1, bias=False)
        self.fc2.weight.data.fill_(1.)
        self.fc3 = torch.nn.Linear(1, 1, bias=False)
        self.fc3.weight.data.fill_(1.)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

    def get_fc1_grad_abs_value(self):
        return torch.abs(self.fc1.weight.grad)

def get_fc1_grad_abs_value(net, x):
    output = net.forward(x)
    output.backward()  # no loss function. Pretending that we want to minimize output
                       # In our case output is scalar, so we can calculate backward
    fc1_grad = net.get_fc1_grad_abs_value().item()
    net.zero_grad()
    return fc1_grad

dict_activation =  {'ELU': torch.nn.ELU(), 'Hardtanh': torch.nn.Hardtanh(),
               'LeakyReLU': torch.nn.LeakyReLU(), 'LogSigmoid': torch.nn.LogSigmoid(),
               'PReLU': torch.nn.PReLU(), 'ReLU': torch.nn.ReLU(), 'ReLU6': torch.nn.ReLU6(),
               'RReLU': torch.nn.RReLU(), 'SELU': torch.nn.SELU(), 'CELU': torch.nn.CELU(),
               'Sigmoid': torch.nn.Sigmoid(), 'Softplus': torch.nn.Softplus(),
               'Softshrink': torch.nn.Softshrink(), 'Softsign': torch.nn.Softsign(),
               'Tanh': torch.nn.Tanh(), 'Tanhshrink': torch.nn.Tanhshrink(),
               'Hardshrink': torch.nn.Hardshrink()}
                           # torch.nn.MultiheadAttention(1,1)
result= []    
              
for name_of_activation in dict_activation:
  
	net = SimpleNet(activation=dict_activation[name_of_activation])

	fc1_grads = []
	for x in torch.randn((NUMBER_OF_EXPERIMENTS, 1)):
	    fc1_grads.append(get_fc1_grad_abs_value(net, x))

	result.append(np.mean(fc1_grads))

names = [key for key in dict_activation]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.barh(names, result)

plt.xlabel("Значение градиента на первом слое")
plt.ylabel("Функция активации")

ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
fig.set_figwidth(15)    #  ширина Figure
fig.set_figheight(6)    #  высота Figure

plt.show()



