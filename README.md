# crelu-pytorch

Implementation of the CReLU activation function from the paper "Loss of Plasticity in Continual Deep Reinforcement Learning".
The CReLU activation function is defined as the concatenation of ReLU with the ReLU of the negative input features:

$$\text{CReLU} = \Big[\text{ReLU}(x), \text{ReLU}(-x) \Big]~.$$

## Installation

```bash
pip install .
```
