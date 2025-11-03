import numpy as np

# Base Layer Class
class Layer:
    def __init__(self):
        pass
    
    def forward_propagation(self, input_data):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

# Fully Connected Layer
class FCLayer(Layer):
    def __init__(self, input_size, output_size, weights=None, biases=None):
        # Xavier Initialization
        self.weights = weights if weights is not None else np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        self.biases = biases if biases is not None else np.zeros((1, output_size), dtype=np.float64)

    def forward_propagation(self, input):
        if input.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Shape mismatch: Input shape {input.shape} does not match expected shape ({input.shape[0]}, {self.weights.shape[0]})")
        self.input = input
        self.output = self.input.dot(self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error, lr):
        input_error = output_error.dot(self.weights.T)
        weights_error = self.input.T.dot(output_error)
        self.weights -= lr * weights_error
        self.biases -= lr * np.sum(output_error, axis=0)
        return input_error

# Activation Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_error, lr):
        return self.activation_derivative(self.input) * output_error

# Loss Function Wrapper 
# User can specify the loss function
class Loss:
    def __init__(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

# Neural Network
class Network:
    def __init__(self, loss, loss_derivative):
        self.layers = []
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, input):
        result = input
        for layer in self.layers:
            result = layer.forward_propagation(result)
        return result
    
    def fit(self, x_train, y_train, x_val, y_val, epochs, lr):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            # Forward propagation
            output = x_train
            for layer in self.layers:
                output = layer.forward_propagation(output)

            train_loss = np.mean(self.loss(y_train, output))
            train_losses.append(train_loss)
            
            # Backpropagation
            error = self.loss_derivative(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, lr)

            val_output = x_val
            for layer in self.layers:
                val_output = layer.forward_propagation(val_output)
            
            val_loss = np.mean(self.loss(y_val, val_output))
            val_losses.append(val_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        
        return train_losses, val_losses

