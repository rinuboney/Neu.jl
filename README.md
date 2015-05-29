# Neu.jl
Neural Networks for Julia

# Usage
Here is an example of a network that learns the XOR function

```julia
# Training data
X_train = [0 0; 0 1; 1 0; 1 1]
Y_train = [0; 1; 1; 0]

# Initialize weights
Theta = init_theta(2, 1, [2])

# Train
Theta, J_list = fit(X_train, Y_train, Theta , 1, 1000)

# Predict
predict(X_train, Theta)
```

This produces the output,
```julia
4x1 Array{Float64,2}:
0.00802484
0.994124  
0.994123  
0.00623008
```

Learning successful!



