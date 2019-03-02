# CPEG586-Assignment3

The neural network was implemented through a class Model:

  `Model(x_inputs, layers, number_epochs=10, batch_size=1, stochastic=False,lr=0.1))`
  
  this class has two methods:
    `fit(X,Y)` and `predict(X)` which returns an array of Y predictions.
    
   The model is dependendant on a class called Layer:
   
    `Layer(neurons, inputs, activationf='Relu')`
    
    this class has three methods:
      `forward(prev_activation)` 
      `backprop(dA_prev)`
      `update_parameters(learning_rate)`

With this structure we can create either a Stochastic Gradient Descent, Gradient Descent or MiniBatch Gradient Descent, 
with any number of layers and activation types.
