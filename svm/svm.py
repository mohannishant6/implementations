import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import cross_val_score
import numpy.random
from sklearn import svm
from mpl_toolkits import mplot3d
from matplotlib import cm

def plotData(X,y):
  # plots the data points with o for the positive examples and x for the negative examples. output is saved to file graph.png
  fig, ax = plt.subplots(figsize=(12,8))
  
  positive = y>0
  negative = y<0
  ax.scatter(X[positive,0], X[positive,1], c='b', marker='o', label='Healthy')
  ax.scatter(X[negative,0], X[negative,1], c='r', marker='x', label='Not Healthy')

  ax.legend()
  ax.set_xlabel('Test 1')
  ax.set_ylabel('Test 2')  
  fig.savefig('graph.png') 
  
  
def predict(X,theta):
  n=X.shape[-1]
  try:
    test=X.shape[1]
  except:
    X=X.reshape(-1,n)

  pred=np.sign(np.sum(np.multiply(theta,X),axis=1))

  return pred

def computeCost(X, y, theta, lambd):
  # function calculates the cost J(theta) and returns its value
  (m,n)=X.shape
  
  temp1=1-np.multiply(y,np.sum(np.multiply(theta,X),axis=1))
  temp2=np.sum(np.maximum(np.zeros(temp1.shape),temp1))/m
  cost=temp2+np.sum(lambd*theta*theta)
   
  return cost

def computeGradient(X,y,theta,lambd):
  # calculate the gradient of J(theta) and return its value
  
  (m,n)=X.shape
    
  ythetax=(np.multiply(y,np.sum(np.multiply(theta,X),axis=1))<=np.ones(m)).reshape(-1,1)
  
  temp2=np.sum(np.multiply(y.reshape(-1,1),np.multiply(X,ythetax)),axis=0)/m
  
  grad=2*lambd*theta-temp2
  
  
  return grad

def gradientDescent(X, y, theta,lambd):
  # iteratively update parameter vector theta
  
  # initialize variables for learning rate and iterations
  alpha = .01
  iters = 5000
  cost = np.zeros(iters)
  (m,n)=X.shape
  theta= np.zeros(n)
  # let's use a reasonable starting point for theta (the value we obtained in the logistic regression assignment)
  #theta=[3.08,2.97,3.69,-5.36]
  
  for i in range(iters):
    theta = theta - alpha * computeGradient(X,y,theta,lambd)
    cost[i] = computeCost(X, y, theta,lambd)
  print(theta)
  return theta, cost

def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale, scale)

def addQuadraticFeature(X):
  # Given feature vector [x_1,x_2] as input, extend this to
  # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
  
  X=np.append(X,(X[:,0]*X[:,0]).reshape(-1,1),axis=1)
  
  return X

def computeScore(X,y,preds):
  # for training data X,y it calculates the number of correct predictions made by the model
  score = sum(y == preds)  
  return score

def plotDecisionBoundary(Xt,y,Xscale,theta):
  # plots the training data plus the decision boundary in the model
  fig, ax = plt.subplots(figsize=(12,8))
  # plot the data
  positive = y>0
  negative = y<0
  ax.scatter(Xt[positive,1]*Xscale[1], Xt[positive,2]*Xscale[2], c='b', marker='o', label='Healthy')
  ax.scatter(Xt[negative,1]*Xscale[1], Xt[negative,2]*Xscale[2], c='r', marker='x', label='Not Healthy')
  # calc the decision boundary
  x=np.linspace(Xt[:,2].min()*Xscale[2],Xt[:,2].max()*Xscale[2],50)
  if (len(theta) == 3):
    # linear boundary
    x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1])/theta[2]*Xscale[2]
  else:
    # quadratic boundary
    x2 = -(theta[0]/Xscale[0]+theta[1]*x/Xscale[1]+theta[3]*np.square(x)/Xscale[3])/theta[2]*Xscale[2]
  # and plot it
  ax.plot(x,x2,label='Decision boundary')
  ax.legend()
  ax.set_xlabel('Test 1')
  ax.set_ylabel('Test 2')  
  fig.savefig('pred.png') 

def plotDecisionBoundary2(Xt,y,Xscale,model):
  # plots the training data plus the decision boundary when using a kernel SVM
  fig, ax = plt.subplots(figsize=(12,8))
  # plot the data
  positive = y>0
  negative = y<0
  ax.scatter(Xt[positive,1]*Xscale[1], Xt[positive,2]*Xscale[2], c='b', marker='o', label='Healthy')
  ax.scatter(Xt[negative,1]*Xscale[1], Xt[negative,2]*Xscale[2], c='r', marker='x', label='Not Healthy')
  # calc the decision boundary
  x_min, x_max = Xt.min() - 0.1, Xt.max() + 0.1
  y_min, y_max = y.min() - 0.1, y.max() + 0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
  Z = model.predict(np.column_stack((np.ones(xx.ravel().shape),xx.ravel(), yy.ravel(), np.square(xx.ravel()))))
  Z = Z.reshape(xx.shape)
  ax.contour(xx*Xscale[1], yy*Xscale[2], Z)
  ax.legend()
  ax.set_xlabel('Test 1')
  ax.set_ylabel('Test 2')  
  fig.savefig('pred2.png')
  


def cross_val(Xt,y,gammas,Cs):
    m=gammas.shape[0]
    n=Cs.shape[0]
    
    accuracies=np.zeros([m,n])
    
    for i in range(0,m):
        for j in range(0,n):
            model = svm.SVC(C=Cs[j], gamma=gammas[i],kernel='rbf')
            model.fit(Xt,y)
            accuracies[i][j]=np.mean(cross_val_score(model,Xt,y,cv=5))
            
    return accuracies
    
    
def plot2D(gamma,C,acc):

    # Define x and y labels
    x = C
    y = gamma
    z = acc

    # Setup figure and colorbar
    fig = plt.figure(figsize=(6,6.5))

    # Define custom grid for image
    grid = ImageGrid(fig, 111,nrows_ncols = (1, 1),cbar_mode='single')

    # Set axis labels
    grid[0].set_xlabel('C')
    grid[0].set_ylabel('gamma')

    # Display image contours and colorbar
    im = grid[0].contourf(x, y, z)
    cbar = grid.cbar_axes[0].colorbar(im)

    # Save figure
    fig.savefig('best_parameters.png')
    
    
def main():
  
  # load the training data
  data=np.loadtxt('health.csv', delimiter=",")
  X=data[:,[0,1]]
  y=data[:,2]
  X=addQuadraticFeature(X)  
 
  # and plot it so we can see how it looks 
  plotData(X,y)

  # add a column of ones to input data
  m=len(y)
  Xt = np.column_stack((np.ones((m, 1)), X))
  (m,n)=Xt.shape # m is number of data points, n number of features

  # rescale training data to lie between 0 and 1
  (Xt,Xscale) = normaliseData(Xt)
  
  lambd=0

  print('testing the prediction function ...')
  theta=np.arange(1,n+1)
  print('when x=[1,1,1] and theta is [1,2,3]) predictions = ',predict(np.ones(n),theta))
  print('when x=[-1,-1,-1] and theta is [1,2,3]) prediction = ',predict(-np.ones(n),theta))
  print('approx expected predictions are 1 and -1')
  input('Press Enter to continue...')
  print('testing the cost function ...')
  theta=np.zeros(n)
  print('when theta is zero cost = ',computeCost(Xt,y,theta,lambd))
  print('approx expected cost value is 1.0')
  input('Press Enter to continue...')

  # calculate the gradient when theta is zero
  print('testing the gradient function ...')
  print('when theta is zero gradient = ',computeGradient(Xt,y,theta,lambd))
  print('approx expected gradient value is [-0.049,-0.074,-0.099,0.086]')
  input('Press Enter to continue...')
   
  # perform gradient descent to "fit" the model parameters
  print('running gradient descent ...')
  theta, cost = gradientDescent(Xt, y, theta,lambd)
  
  print('after running gradientDescent() theta=',theta)
  print('approx expected value is [3.08,2.97,3.69,-5.36], or values with about the same ratio')
 
  # plot the prediction
  plotDecisionBoundary(Xt,y,Xscale,theta)

  preds = predict(Xt,theta)
  score = computeScore(Xt,y,preds)
  print('accuracy = {0:.2f}%'.format(score/len(y)*100))
  print('expected value is about 78%')

  # plot how the cost varies as the gradient descent proceeds
  fig2, ax2 = plt.subplots(figsize=(12, 8))
  ax2.semilogy(cost,'r')
  ax2.set_xlabel('iteration')
  ax2.set_ylabel('cost')
  fig2.savefig('cost.png')
  
  # Using sklearn to train the SVM
  
  input('Press Enter to continue...')
  print('Testing accuracy from sklearn...')
  model = svm.SVC(C=1,kernel='linear')
  model.fit(Xt,y)
  theta=np.concatenate((model.intercept_,(model.coef_.ravel())[1:4]))
  cost=[1,2] # to avoid plotting error
  
  preds = predict(Xt,theta)
  score = computeScore(Xt,y,preds)
  print('accuracy = {0:.2f}%'.format(score/len(y)*100))
  
  
  # Using kernels with an SVM
  
  input('Press Enter to continue...')
  print('Testing accuracy from sklearn using kernel...')
   
  model = svm.SVC(C=0.5, gamma=0.75,kernel='rbf')
  model.fit(Xt,y)
  plotDecisionBoundary2(Xt,y,Xscale,model)
  preds = model.predict(Xt)
  score = computeScore(Xt,y,preds)
  print('accuracy = {0:.2f}%'.format(score/len(y)*100))


  # Cross Validation
  
  input('Press Enter to continue...')
  print('Selecting C and gamma (𝜆 and σ) using cross-validation...')
  
  # get accuracies for gammas and Cs ranging from 0.001 upto 10 
  gammas=np.linspace(0.001,10,20)
  Cs=np.linspace(0.001,10,20)
  accuracies=cross_val(Xt,y,gammas,Cs)
  
  # get the contour plot to determine best parameters
  plot2D(gammas,Cs,accuracies)

  print('From best_parameters.png we can get optimum values of C and gamma.')


if __name__ == '__main__':
  main()

