import sys
sys.dont_write_bytecode = True

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Anomaly_Net(nn.Module):
	def __init__(self):
		super(Anomaly_Net, self).__init__()
		self.layer1 = nn.Linear(65536, 100)
		self.layer2 = nn.Linear(100, 50)
		self.layer3 = nn.Linear(50, 100)
		self.layer4 = nn.Linear(100, 65536)

		# initialize weight with glorot uniform 
		nn.init.zeros_(self.layer1.bias)
		nn.init.xavier_uniform_(self.layer2.weight) 
		nn.init.zeros_(self.layer2.bias)
		nn.init.xavier_uniform_(self.layer3.weight) 
		nn.init.zeros_(self.layer3.bias)
		nn.init.xavier_uniform_(self.layer4.weight) 
		nn.init.zeros_(self.layer4.bias)

	def forward(self, x):
		z = T.tanh(self.layer1(x))
		z = T.tanh(self.layer2(z))
		z = T.tanh(self.layer3(z))
		# output layer use sigmoid?
		z = T.tanh(self.layer4(z))

		return z