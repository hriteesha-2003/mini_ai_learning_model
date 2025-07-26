import numpy as np # type: ignore
x=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_dict={
    "AND":np.array([[0],[0],[0],[1]]),
    "OR": np.array([[0],[1],[1],[1]]),
    "NAND": np.array([[1],[1],[1],[0]]),
    "NOR": np.array([[1],[0],[0],[0]]),
    "XOR": np.array([[0],[1],[1],[0]]),
    "XNOR": np.array([[1],[0],[0],[1]])
}
