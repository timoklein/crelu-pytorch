# crelu-pytorch

Implementation of the CReLU activation function from the paper "Loss of Plasticity in Continual Deep Reinforcement Learning".
The CReLU activation function is defined as the concatenation of ReLU with the ReLU of the negative input features:

$$\text{CReLU} = \Big[\text{ReLU}(x), \text{ReLU}(-x) \Big]~.$$

Note that CReLU is only 0 when an input feature is exactly 0.
Compared to ReLU, CReLU doubles the number of features in the output. The CReLU activation function is used in the paper to
mitigate plasticity loss.

## Installation

```bash
pip install .
```
