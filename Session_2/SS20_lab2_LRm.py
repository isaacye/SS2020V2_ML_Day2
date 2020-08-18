import matplotlib.pyplot as plt
import numpy as np

X = list(range(10))
Y = [1, 1, 2, 4, 5, 7, 8, 9, 10, 9]

plt.scatter(X,Y)
plt.savefig('datascatter.png')
plt.close()

class H():
  
    def __init__(self, w): 
        self.w = w
    
    def forward(self, x):
        return self.w * x     # H(x) = Wx

"""## **Cost(Loss) Function**"""

def cost(h, X, Y):
    error = 0
    for i in range(len(X)):
        error += (h.forward(X[i]) - Y[i])**2
    error = error / len(X)
    return error

"""##**Checking cost function**##"""

list_w = []
list_c = []
for i in range(-20, 20):
    w = i * 0.5
    h = H(w)
    c = cost(h, X, Y)
    list_w.append(w)
    list_c.append(c)
    
print(list_w)
print(list_c) 

plt.figure(figsize=(10,5))
plt.xlabel('w')
plt.ylabel('cost')
plt.scatter(list_w, list_c, s=3)
plt.savefig('result0.png')
plt.close()

"""##**Gradient Decent**"""

def cal_grad(w, cost): 
    h = H(w)
    cost1 = cost(h, X, Y)
    eps = 0.00001 
    h = H(w+eps) 
    cost2 = cost(h, X, Y)
    dcost = cost2 - cost1
    dw = eps
    grad = dcost / dw
    return grad, (cost1+cost2)*0.5

def cal_grad2(w, cost): 
    h = H(w)
    grad = 0
    for i in range(len(X)):
        grad += 2 * (h.forward(X[i]) - Y[i]) * X[i]
    grad = grad / len(X)
    c = cost(h, X, Y)
    return grad, c

"""##**Running case: Testing two different gradient decent models**"""

# ===== Initializing weights (w) ==== #
w1 = 1.4
w2 = 1.4

# ===== Learning rate ===== #

lr = 0.01

list_w1 = []
list_c1 = []
list_w2 = []
list_c2 = []

for i in range(100): 
    grad, mean_cost = cal_grad(w1, cost)
    grad2, mean_cost2 = cal_grad2(w2, cost)

    w1 -= lr * grad
    w2 -= lr * grad2
    list_w1.append(w1)
    list_w2.append(w2)
    list_c1.append(mean_cost)
    list_c2.append(mean_cost2)
    
print(w1, mean_cost, w2, mean_cost2) 

plt.scatter(list_w1, list_c1, label='analytic', marker='*')
plt.plot(list_w2, list_c2, label='formula')
plt.legend()
plt.savefig('result1.png')
plt.close()

"""##**Checking results: Best fit**"""

x_grid=np.arange(10)
pred_y1 = w1*x_grid
plt.plot(X,pred_y1)
plt.scatter(X,Y)
#plt.show()
plt.savefig('result2.png')
plt.close()
