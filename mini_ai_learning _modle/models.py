import pickle
import numpy as np # type: ignore
import data, utils
from data import x, y_dict
from utils import sigmoid, sigmoid_derivative


def train_gate(gate_name, x, y, lr=0.1, epochs=10000):
    print(f"Training {gate_name} gate...")
    np.random.seed(2)
    weights = 2*np.random.random((2, 1))-1
    bias = 0.0
    for epoch in range(epochs):
        z=np.dot(x,weights)+bias
        output=sigmoid(z)

        error = y - output

        delta_weights = lr * np.dot(x.T, error * sigmoid_derivative(output))
        delta_bias = lr * np.sum(error * sigmoid_derivative(output))

        weights += delta_weights
        bias += delta_bias
    return weights, bias
           
def gate_xor_xnor(gate_name,x,y,lr=0.1,epochs=50001):
    print(f"Training {gate_name} gate...")
    np.random.seed(1)
    
    input_size = 2;
    hidden_size = 2;
    output_size = 1;
    
    w1 = 2 * np.random.random((input_size, hidden_size)) - 1
    b1 = np.zeros((1, hidden_size))
    w2 = 2 * np.random.random((hidden_size, output_size)) - 1
    b2 = np.zeros((1, output_size))

    y = y.reshape(-1, 1)

    for epoch in range(1,epochs):
        z1 = np.dot(x,w1)+b1
        output1=sigmoid(z1)

        z2=np.dot(output1,w2)+b2
        output2=sigmoid(z2)

        error=y-output2

        delta2 = error*sigmoid_derivative(output2)
        delta1 = delta2.dot(w2.T) * sigmoid_derivative(output1)

        w1+= lr* np.dot(x.T,delta1)
        b1+= lr*np.sum(delta1, axis=0, keepdims=True)
        w2+= lr * np.dot(output1.T, delta2)
        b2+= lr * np.sum(delta2, axis=0, keepdims=True)


    print(f"\nFinal output for {gate_name}:")

    preds = sigmoid(np.dot(sigmoid(np.dot(x, w1) + b1), w2) + b2)

    for i in range(len(x)):
        print(f"Input: {x[i]} → Predicted: {preds[i][0]:.4f}, Expected: {y[i][0]}")
    print("Training completed for", gate_name)
    print("-" * 50)


    #for xnor gate and xor gate
    model = {
        'weights1': w1,
        'bias1': b1,
        'weights2': w2,
        'bias2': b2,        
        'gate_name': gate_name        
    }
    with open(f"{gate_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model for {gate_name} gate saved as {gate_name}_model.pkl")
    print("-" * 50)


    combined_model = {
        'weights1': w1,
        'bias1': b1,
        'weights2': w2,
        'bias2': b2,
        'gate_name': gate_name
    }
    with open(f"{gate_name}_model.pkl", "wb") as f:
        pickle.dump(combined_model, f)
    print(f"Model for {gate_name} gate saved as {gate_name}_model.pkl")
    print("-" * 50)