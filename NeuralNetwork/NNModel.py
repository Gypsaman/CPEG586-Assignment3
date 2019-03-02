import numpy as np
import math

def relu(data):
    sh = data.shape
    result = np.array([[x if x > 0 else 0 for x in row] for row in data]).reshape(sh)
    return result

def sigmoid(data):
    result = 1/(1+np.exp(-1*data))
    return result

def tanh(data):
    result = (np.exp(data)-np.exp(-1*data)/(np.exp(data)+np.exp-1*(data)))
    return result

def loss(predict,y):
    Loss = (0.5 * (np.multiply((predict-y),(predict-y)))).sum()/predict.shape[1]
    return Loss

class Layer(object):
    """description of class"""
    def __init__(self, neurons, inputs, activationf='Relu'):
        assert(activationf == 'Relu' or activationf == 'Sigmoid') 

        self.layerNodes = neurons
        self.inputNodes = inputs

        self.weights = np.random.uniform(low=-0.1,high=0.1,size=(neurons,inputs))
        self.biases = np.random.uniform(low=-1,high=1,size=(neurons,1))
        self.activationf = activationf
        self.prev_a = np.zeros((inputs,1))
        self.delta = np.zeros((neurons,1))
        self.dw = np.zeros(neurons)
        self.db = np.zeros(neurons)
        self.activation = np.zeros((neurons,1))
       
    
    def forward(self,prev_activation):
        assert prev_activation.shape[0] == self.inputNodes
        self.prev_a = prev_activation.astype(float)
        s = np.dot(self.weights,self.prev_a) + self.biases
        if self.activationf == 'Relu':
            self.activation = relu(s)
        elif self.activationf == 'Sigmoid':
            self.activation = sigmoid(s)
        elif self.activationf == 'Tanh':
            self.activation == tanh(s)
        return self.activation

    def backprop(self,dA,batch,epoch):
        m = self.prev_a.shape[1]
        if self.activationf == 'Relu':
            self.delta = dA*(1)
        elif self.activationf == 'Sigmoid':
            self.delta = dA*(self.activation*(1-self.activation))
        elif self.activationf == 'Tanh':
            self.delta = dA*(1-self.activation*self.activation)
        self.dw = 1/m*np.dot(self.delta,self.prev_a.T)
        self.db = 1/m*np.sum(self.delta,axis=1,keepdims=True)
        da_prev = np.dot(self.weights.T,self.delta)
        return da_prev
        
    def update_parameters(self,lr):
        self.weights -= lr * self.dw
        self.biases -=  lr * self.db


class Model(object):
    """description of class"""
    def __init__(self, x_inputs, layers, number_epochs=10, batch_size=1, stochastic=False,lr=0.1):
        self.layers = []
        self.epochs =  number_epochs
        self.batchsize = batch_size
        self.stochastic = stochastic
        self.lr = lr

        prev_a = x_inputs
        for size,actf in layers:
            layer = Layer(neurons=size,inputs=prev_a,activationf=actf)
            self.layers.append(layer)
            prev_a = size
    
    def fit(self,X,Y):
        self.samples = X.shape[0]
        batches = math.floor(self.samples / self.batchsize)
        if batches * self.batchsize < self.samples:
            batches += 1

        for epoch in range(self.epochs):
            cost = 0
            for batch in range(batches):
                startbatch = batch * self.batchsize
                endbatch = min((batch+1)*self.batchsize,self.samples)
                a_prev = X[:,startbatch:endbatch ]
                y_sample = Y[:,startbatch: endbatch]
                ## Forward Calculate
                for layer in self.layers:
                    a_prev = layer.forward(a_prev)
                cost += loss(a_prev,y_sample)
                # Backward Propagation
                da_prev = -(y_sample-a_prev)
                for layer in reversed(self.layers):
                    da_prev = layer.backprop(da_prev,batch,epoch).astype(float)
                    layer.update_parameters(self.lr)
            print("epoch =" + str(epoch) + " loss = " + str(cost)) 
    
    def predict(self,X):
        a_prev = X
        for layer in self.layers:
            a_prev = layer.forward(a_prev)
        return a_prev






