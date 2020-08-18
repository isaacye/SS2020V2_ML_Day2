import torch
print(torch.__version__)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import time


st = time.time()
"""##**Data Preparation**"""

num_data = 2400
x1 = np.random.rand(num_data) *10
x2 = np.random.rand(num_data) *10
e = np.random.normal(0, 0.5, num_data)
X= np.array([x1,x2]).T  # T for transpose from (2, 2400) to (2400, 2)
y=2*np.sin(x1) + np.log(0.5*x2**2)+e

"""Data split"""

train_X, train_y = X[:1600, :], y[:1600]
val_X, val_y = X[1600:2000, :], y[1600:2000]
test_X, test_y = X[2000:, :], y[2000:]

"""##**Visualizating input data**"""

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1, 3, 1, projection='3d') # size 1 row, 3 col, location 1
ax1.scatter(train_X[:, 0], train_X[:, 1], train_y, c=train_y, cmap='jet')

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('Train Set Distribution')
ax1.set_zlim(-10, 6)  # z axis limit
ax1.view_init(40, -60) #view angle
ax1.invert_xaxis() #direction of number line

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(val_X[:, 0], val_X[:, 1], val_y, c=val_y, cmap='jet')

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('Validation Set Distribution')
ax2.set_zlim(-10, 6)
ax2.view_init(40, -60)
ax2.invert_xaxis()

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('Test Set Distribution')
ax3.set_zlim(-10, 6)
ax3.view_init(40, -60)
ax3.invert_xaxis()

#plt.show()
plt.savefig('checkinginput.png')
plt.close()

"""##**Model(Hypothesis) Define**"""

class MLPModel(nn.Module):
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=200)
        self.linear3 = nn.Linear(in_features=200, out_features=200)
        self.linear4 = nn.Linear(in_features=200, out_features=200)
        self.linear5 = nn.Linear(in_features=200, out_features=200)
        self.linear6 = nn.Linear(in_features=200, out_features=200)
        self.linear7 = nn.Linear(in_features=200, out_features=200)
        self.linear8 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear8(x)
        return x

"""##**Cost(Loss) Function**"""

reg_loss = nn.MSELoss()

"""##**Training & Evaluation**"""

print('Check if GPU is available:{}'.format(torch.cuda.is_available()))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device != 'cpu':
    print('Hello GPU!')
else:
    print('Hello CPU!')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # ====== Model selection ======= #
model = MLPModel()
# 
# # ====== GPU selection ======= #
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
# 
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
# 
lr = 0.005
optimizer = optim.SGD(model.parameters(), lr =lr)  # model.parameters : W, b of linear model
# 
list_epoch = []
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []
# 
epoch = 4000
# 
for i in range(epoch):
#   # ===== Training ===== #
        model.train() #setting mode for model train
        optimizer.zero_grad() # initialize gradient
# 
        input_x = torch.Tensor(train_X)
        true_y = torch.Tensor(train_y)
#  
        if device != 'cpu':
            input_x = input_x.to(device)   # send to GPU
            true_y = true_y.to(device) # send to GPU 
       
        pred_y = model(input_x)
# 
        loss = reg_loss(pred_y.squeeze(), true_y) # dropping column of pred_y dimession
        loss.backward() # backward() calculate gradients
        optimizer.step() # update gradients using step()
        list_epoch.append(i)
# 
        if device != 'cpu':
            list_train_loss.append(loss.cpu().detach().numpy())
        else:
            list_train_loss.append(loss.detach().numpy())
# 
# 
#   # ===== Validation ===== #
        model.eval()
        optimizer.zero_grad()
        input_x = torch.Tensor(val_X)
        true_y = torch.Tensor(val_y)
#    
        if device != 'cpu':
            input_x = input_x.to(device)   # send to GPU
            true_y = true_y.to(device) # send to GPU 
#       
        pred_y = model(input_x)
# 
        loss = reg_loss(pred_y.squeeze(), true_y)
# 
        if device != 'cpu':
            list_val_loss.append(loss.cpu().detach().numpy())
        else:
            list_val_loss.append(loss.detach().numpy())
# 
#   # ====== Evaluation ======= #
#   
        if i % 200 == 0: # evaluate it every 200
#         
#         # ====== Calculate MAE ====== #
            model.eval()
            optimizer.zero_grad()
            input_x = torch.Tensor(test_X)
            true_y = torch.Tensor(test_y)
         
            if device != 'cpu':
                input_x = input_x.to(device)   # send to GPU
                true_y = true_y.to(device) # send to GPU 
                pred_y = model(input_x).cpu().detach().numpy()
                mae = mean_absolute_error(true_y.cpu(), pred_y) 
            else:
                pred_y = model(input_x).detach().numpy()
                mae = mean_absolute_error(true_y, pred_y) 
 
            list_mae.append(mae)
            list_mae_epoch.append(i)
#         
            fig = plt.figure(figsize=(15,5))
#         
#         # ====== True Y Scattering ====== #
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax1.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')
         
            ax1.set_xlabel('x1')
            ax1.set_ylabel('x2')
            ax1.set_zlabel('y')
            ax1.set_zlim(-10, 6)
            ax1.view_init(40, -40)
            ax1.set_title('True test y')
            ax1.invert_xaxis()
 
#         # ====== Predicted Y Scattering ====== #
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax2.scatter(test_X[:, 0], test_X[:, 1], pred_y, c=pred_y[:,0], cmap='jet')
# 
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2')
            ax2.set_zlabel('y')
            ax2.set_zlim(-10, 6)
            ax2.view_init(40, -40)
            ax2.set_title('Predicted test y')
            ax2.invert_xaxis()
# 
#         # ====== Just for Visualizaing with High Resolution ====== #
            input_x = torch.Tensor(train_X)
#         
            if device != 'cpu':
                input_x = input_x.to(device)   # send to GPU
                pred_y = model(input_x).cpu().detach().numpy()
            else:
                pred_y = model(input_x).detach().numpy()
         
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax3.scatter(train_X[:, 0], train_X[:, 1], pred_y, c=pred_y[:,0], cmap='jet')
# 
            ax3.set_xlabel('x1')
            ax3.set_ylabel('x2')
            ax3.set_zlabel('y')
            ax3.set_zlim(-10, 6)
            ax3.view_init(40, -40)
            ax3.set_title('Predicted train y')
            ax3.invert_xaxis()
#         
#         plt.show()
            plt.savefig('result1.png')
            print(i, loss)

plt.close()
"""##**Presenting loss and error**"""

fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_ylim(0, 5)
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_mae_epoch, list_mae, marker='x', label='mae metric')

ax2.set_xlabel('epoch')
ax2.set_ylabel('mae')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs mae')


#plt.show()
plt.savefig('result2.png')
plt.close()

et = time.time()

print(et - st)
