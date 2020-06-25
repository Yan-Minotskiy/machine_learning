import torch
from torch import tensor, log, prod

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
device = torch.device('cuda:0'
                      if torch.cuda.is_available()
                      else 'cpu')
w = w.to(device)
function =  (w + 7).log().log().prod()
function.backward()
                      
#print(w.grad) # Код для самопроверки