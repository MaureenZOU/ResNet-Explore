import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt

input_size = 1
hidden_size1 = 100
hidden_size2 = 200
output_size = 1

batch_size = 32

learning_rate = 2e-4

optim_betas = (0.9, 0.999)

num_epochs = 200
save_epoch = 5

sample_size = 10000


sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)

class AutoEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(AutoEncoder, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size1)
		self.map2 = nn.Linear(hidden_size1, hidden_size2)
		self.map3 = nn.Linear(hidden_size2, hidden_size2)
		self.map4 = nn.Linear(hidden_size2, hidden_size1)
		self.map5 = nn.Linear(hidden_size1, output_size)

	def forward(self, x):
		x = F.relu(self.map1(x))
		x = F.relu(self.map2(x))
		x = F.relu(self.map3(x))
		x = F.sigmoid(self.map4(x))
		return self.map5(x)

class Sampler():
	def __init__(self):
		self.s = np.random.uniform(-40, 40,sample_size)

	def getData(self):
		dataset = []
		for x in self.s:
			dataset.append([x, self.f(x)])
		return np.array(dataset)

	def f(self, x):
		return x + 3*np.sin(x)

def extract(v):
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d), np.std(d)]

#plain MLP
def Train(): 
	sampler = Sampler()
	dataset = sampler.getData()

	Net = AutoEncoder(input_size, output_size)

	criterion_l2 = nn.MSELoss()
	criterion_l1 = nn.L1Loss()
	optimizer = optim.Adam(Net.parameters(), lr = learning_rate, betas = optim_betas)

	converge = []

	for epoch in range(0, num_epochs):
		np.random.seed(epoch)
		np.random.shuffle(dataset)

		for i in range(0, int(len(dataset)/batch_size)):
			cur_sample = dataset[i*batch_size:(i+1)*batch_size]
			x = cur_sample[:,0]
			y = cur_sample[:,1]

			x = np.reshape(x, (x.shape[0], 1))
			y = np.reshape(y, (y.shape[0], 1))

			cur_x = Variable(torch.Tensor(x))
			cur_y = Variable(torch.Tensor(y))

			optimizer.zero_grad()
			
			pre_y = Net.forward(cur_x)

			loss = criterion_l2(pre_y, cur_y)

			loss.backward()
			optimizer.step()
		
		print("Epoch:" + str(epoch) + "   current Loss: " + str(extract(loss)[0]))
		converge.append([epoch, extract(loss)[0]])

		if epoch % save_epoch == 0:
			torch.save(Net.state_dict(), "./orinal/model/epoch" + str(epoch))

			x = dataset[0:500][:,0]
			y = dataset[0:500][:,1]

			test_x = np.reshape(x, (x.shape[0], 1))

			test_x = Variable(torch.Tensor(test_x))
			pre_y = Net.forward(test_x)
			pre_y = extract(pre_y)

			plt.plot(x, y, "r.")
			plt.plot(x, pre_y, "g.")

			plt.savefig("./orinal/image/"+str(epoch)+".png")
			plt.clf()

	con_x = np.array(converge)[:, 0]
	con_y = np.array(converge)[:, 1]

	plt.plot(con_x, con_y, "r")
	plt.savefig("./orinal/converge/converge.png")

#identity pretrain
def Train_pre():
	sampler = Sampler()
	dataset = sampler.getData()

	Net = AutoEncoder(input_size, output_size)
	Net.load_state_dict(torch.load("/Users/maureen/Documents/Python/identity_pretrain/identity/model/epoch100"))

	criterion_l2 = nn.MSELoss()
	criterion_l1 = nn.L1Loss()
	optimizer = optim.Adam(Net.parameters(), lr = learning_rate, betas = optim_betas)

	converge = []

	for epoch in range(0, num_epochs):
		np.random.seed(epoch)
		np.random.shuffle(dataset)

		for i in range(0, int(len(dataset)/batch_size)):
			cur_sample = dataset[i*batch_size:(i+1)*batch_size]
			x = cur_sample[:,0]
			y = cur_sample[:,1]

			x = np.reshape(x, (x.shape[0], 1))
			y = np.reshape(y, (y.shape[0], 1))

			cur_x = Variable(torch.Tensor(x))
			cur_y = Variable(torch.Tensor(y))

			optimizer.zero_grad()
			
			pre_y = Net.forward(cur_x)

			loss = criterion_l2(pre_y, cur_y)

			loss.backward()
			optimizer.step()
		
		print("Epoch:" + str(epoch) + "   current Loss: " + str(extract(loss)[0]))
		converge.append([epoch, extract(loss)[0]])

		if epoch % save_epoch == 0:
			torch.save(Net.state_dict(), "./pre_identity/model/epoch" + str(epoch))

			x = dataset[0:500][:,0]
			y = dataset[0:500][:,1]

			test_x = np.reshape(x, (x.shape[0], 1))

			test_x = Variable(torch.Tensor(test_x))
			pre_y = Net.forward(test_x)
			pre_y = extract(pre_y)

			plt.plot(x, y, "r.")
			plt.plot(x, pre_y, "g.")

			plt.savefig("./pre_identity/image/"+str(epoch)+".png")
			plt.clf()

	con_x = np.array(converge)[:, 0]
	con_y = np.array(converge)[:, 1]

	plt.plot(con_x, con_y, "r")
	plt.savefig("./pre_identity/converge/converge.png")

#resnet MLP
def Train_res():
	sampler = Sampler()
	dataset = sampler.getData()

	Net = AutoEncoder(input_size, output_size)

	criterion_l2 = nn.MSELoss()
	criterion_l1 = nn.L1Loss()
	optimizer = optim.Adam(Net.parameters(), lr = learning_rate, betas = optim_betas)

	converge = []

	for epoch in range(0, num_epochs):
		np.random.seed(epoch)
		np.random.shuffle(dataset)

		for i in range(0, int(len(dataset)/batch_size)):
			cur_sample = dataset[i*batch_size:(i+1)*batch_size]
			x = cur_sample[:,0]
			y = cur_sample[:,1]

			x = np.reshape(x, (x.shape[0], 1))
			y = np.reshape(y, (y.shape[0], 1))

			cur_x = Variable(torch.Tensor(x))
			cur_y = Variable(torch.Tensor(y))

			optimizer.zero_grad()
			
			pre_y = Net.forward(cur_x)

			loss = criterion_l2(pre_y, cur_y + cur_x)

			loss.backward()
			optimizer.step()
		
		print("Epoch:" + str(epoch) + "   current Loss: " + str(extract(loss)[0]))
		converge.append([epoch, extract(loss)[0]])

		if epoch % save_epoch == 0:
			torch.save(Net.state_dict(), "./resnet_sqr/model/epoch" + str(epoch))

			x = dataset[0:500][:,0]
			y = dataset[0:500][:,1]

			test_x = np.reshape(x, (x.shape[0], 1))

			test_x = Variable(torch.Tensor(test_x))
			pre_y = Net.forward(test_x) - test_x
			pre_y = extract(pre_y)

			plt.plot(x, y, "r.")
			plt.plot(x, pre_y, "g.")

			plt.savefig("./resnet_sqr/image/"+str(epoch)+".png")
			plt.clf()

	con_x = np.array(converge)[:, 0]
	con_y = np.array(converge)[:, 1]

	plt.plot(con_x, con_y, "r")
	plt.savefig("./resnet_sqr/converge/converge.png")


def Identity():
	sampler = Sampler()
	dataset = sampler.getData()

	Net = AutoEncoder(input_size, output_size)
	criterion_l2 = nn.MSELoss()
	criterion_l1 = nn.L1Loss()
	optimizer = optim.Adam(Net.parameters(), lr = learning_rate, betas = optim_betas)

	converge = []

	for epoch in range(0, num_epochs):
		np.random.shuffle(dataset)

		for i in range(0, int( len(dataset)/batch_size)):
			cur_sample = dataset[i*batch_size:(i+1)*batch_size]
			x = cur_sample[:,0]
			y = cur_sample[:,1]

			x = np.reshape(x, (x.shape[0], 1))
			y = np.reshape(y, (y.shape[0], 1))

			cur_x = Variable(torch.Tensor(x))
			cur_y = Variable(torch.Tensor(y))

			optimizer.zero_grad()
			
			pre_y = Net.forward(cur_x)

			loss = criterion_l1(pre_y, cur_x)

			loss.backward()
			optimizer.step()
		
		print("Epoch:" + str(epoch) + "   current Loss: " + str(extract(loss)[0]))
		converge.append([epoch, extract(loss)[0]])
		if epoch % save_epoch == 0:
			torch.save(Net.state_dict(), "./identity/model/epoch" + str(epoch))


Train_res() #please edit this function


