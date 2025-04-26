# Diagram

<div style="display: inline-block; margin-right: 10px;">
    <img src="./Images/LSTM Neural Network-Sigmoid Node.drawio.png" width="200" height="200">
</div>
<div style="display: inline-block;">
    <img src="./Images/LSTM Neural Network-Tanh Node.drawio.png" width="200" height="200">
</div>

<img src=".\Images\LSTM Neural Network-LTSM - Gates.drawio.png" width="300" height="200">

<img src=".\Images\LSTM Neural Network-LTSM.drawio.png" width="300" height="200">

<img src=".\Images\LSTM Neural Network-LTSM Notation.drawio.png" width="300" height="200">

# Definitions

First, we define the sum as: $u_{kt} = \sum_{i=1}^N W_{ki}x_{ti} +W_{kh}h_{t-1}+b_k $ where:
1. $k$ is the letter belonging to the node
2. $t$ is the sequence step
3. $N$ is the number of features

 This makes notation much easier to read.

1. Forget Gate: $f_t = \sigma(u_{ft})$
2. Input Gate: $I_t = \sigma(u_{It})$
3. Candidate Cell State: $s_t = \tanh(u_{st})$
4. Output Gate: $o_t = \sigma(u_{ot})$

Long Term Memory will be referred to as $c_t$ and Short Term Memory will be referred to as $h_t$

# Forward Propagation 

1. Long Term Memory Update: $c_t = f_t \times c_{t-1} + I_t \times s_t$
2. Short Term Memory Update: $h_t = o_t \times \tanh(c_t)$

# Backward Propagation 

Let's take the simple case, where we are training a sequence to one. We only compare the final output to the ground truth.

```math
\begin{align*}
L &= l(y_t, \hat{y}_t)\\
 &= l(y_t, h_t)
\end{align*}
```

Since we want to calculate the differential of the loss with respect to the weights, we need to do this for each of the gates due to the fact each gradient passes by a different path. 

Since we are just dealing with a linear sum, if we find the derivative of the loss with respect to one of the weights, then this will be correct for all of the weights for the given node.

```math
\begin{align*}
\frac{dL}{dW_{oi}} &= \frac{dL}{h_t} \frac{h_t}{o_t} \frac{o_t}{u_{ot}} \frac{u_{ot}}{dW_{oi}} \\
\frac{dL}{dW_{fi}} &= \frac{dL}{h_t} \frac{h_t}{c_t} \frac{c_t}{f_t} \frac{f_t}{u_{ft}}\frac{u_{ft}}{dW_{fi}}\\
\frac{dL}{dW_{Ii}} &= \frac{dL}{h_t} \frac{h_t}{c_t} \frac{c_t}{I_t} \frac{I_t}{u_{It}}\frac{u_{It}}{dW_{Ii}}
\end{align*}
```

However, there is one big issue. Calculating their respective $u_{kt}$ with respect to the weight. This is due to the recursive nature of the formula and is what motivates backpropogation through time.

# Illustrative example of the problem of calculating the derivative




```math
\begin{align*}
L &= l(y_T-\hat{y}_T)^2 \\
 &= (y_T-h_T)^2
\end{align*}
```