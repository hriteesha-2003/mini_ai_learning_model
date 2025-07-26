import numpy as np # type: ignore
import data, utils
from data import x, y_dict
from utils import sigmoid, sigmoid_derivative
from models import train_gate, gate_xor_xnor
import pickle

for gate in y_dict.keys():
    y = y_dict[gate]

    if gate in ["XOR", "XNOR"]:
        gate_xor_xnor(gate, x, y) 
    else:
        weights, bias = train_gate(gate, x, y)
        final_output = sigmoid(np.dot(x, weights) + bias)
        # final_output = np.round(final_output)
        print(f"\nFinal weights for {gate} gate: {weights.flatten()}")
        print(f"Final bias for {gate} gate: {bias}")
        print(f"Final output for {gate} gate: {final_output.flatten()}")
        print(f"Expected output for {gate} gate: {y.flatten()}")
        print(f"Training completed for {gate}")
        print("-" * 50)

       
        model = {
            'weights': weights,
            'bias': bias,
            'gate_name': gate
        }
        with open(f"{gate}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Model for {gate} gate saved as {gate}_model.pkl")
        print("-" * 50)