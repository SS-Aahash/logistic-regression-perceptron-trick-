from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
x, y = make_classification(n_samples=100, n_features=2,n_informative=1,n_redundant=0,random_state=41,n_classes=2,n_clusters_per_class=1) #just making the random data sets for out model
# x is the input matrix(like cgpa,iq) & y is the output matrix(like placed or not)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def perceptron(x,y):
    x = np.insert(x,0,1,axis=1) # we are inseting column of '1s' in the "x,y" matrix, axis=1 means that insertion should happen at x axis(vertically) not horizontally or y_axis, 0 is the index before which column 1's will be inserted
    weights = np.ones(x.shape[1])# initially the weights [Wo,W1,W2] are assigned to 1
    lr = 0.6
    for i in range(1000):  
        j = np.random.randint(0,100) # as there are 100 rows
        # y_hat = 1 if np.dot(x[j],weights) > 0 else 0 # predicted value with step function, citation 1
        y_hat = sigmoid(np.dot(x[j],weights)) # predicted value, with sigmoid function(slightly better)
        weights = weights + lr*(y[j]-y_hat)*x[j] # citation 2
    return weights[0],weights[1:]

intercept, coef = perceptron(x,y)

#citation 3
m = -(coef[0]/coef[1])
c = -(intercept/coef[1])

x_input = np.linspace(-3,3,100)
y_input = m*x_input+c

plt.plot(x_input,y_input,color='red',linewidth=3) # this line of code makes the line on the graph

# x[:,0], ':' indicates that we want to select all the rows of the array x, and 0 is the column number, in summary x[:,0] means 'give me all the elements from the first column of the array x'
# c = color, in this context it is used to specify the color of each marker based on the values of y
# s is the size of the scatter point
plt.scatter(x[:,0],x[:,1],c=y,s=100) # this line of code makes all the scatter points

# ylim is used to set the range of values that will be displayed along the y-axis
plt.ylim(-3,2)
plt.show()