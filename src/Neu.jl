function sigmoid(z)
  1 ./ (1 .+ e .^ (-z))
end

function sigmoid_grad(z)
  sigmoid(z) .* (1 - sigmoid(z))
end

function init_theta(num_input, num_output, nums_hidden)
  num_units = [num_input, nums_hidden..., num_output]
  layers = length(num_units)
  Theta = [rand(num_units[i], num_units[i-1]+1) - 0.5 for i = 2:layers]
end

function cost(Y, Y_pred)
  len = size(Y, 1)
 J = (-1.0 / len ) * sum((Y .* log(Y_pred)) + ((1-Y) .* log(1-Y_pred)))
end

function forward_prop(X, Theta)
  A = {}
  H = {}
  push!(H, X)
  push!(A, zeros(1, 1))

  for t=1:length(Theta)
    # add bias
    H[t] = [ones(size(H[t], 1), 1) H[t]]
    # pre-activation
    push!(A, H[t] * Theta[t]')
    # activation
    push!(H, sigmoid(A[t+1]))
  end
  A, H, H[end]
end

function back_prop(X, Y, Theta)
  len = size(X, 1)
  T = length(Theta)

  Theta_grad = {}
  for i=1:T
    push!(Theta_grad, zeros(size(Theta[i])))
  end

  A, H, Y_pred = forward_prop(X, Theta)

  delta_N = {}
  
  for t=1:T
    push!(delta_N, {})
  end

  delta = Y_pred - Y

  delta_N[T] = delta

  for t=0:T-2
    delta = (delta*Theta[T-t][:, 2:end]) .* sigmoid_grad(A[T-t])
    delta_N[T-t-1] = delta
  end

  for t=1:T
    Theta_grad[t] = Theta_grad[t] + delta_N[t]' * H[t]
  end
  
  Y_pred, Theta_grad
end

function fit(X, Y, Theta, learning_rate, epochs)
  J_list = zeros(epochs)
  for i=1:epochs
    Y_pred, Theta_grad = back_prop(X, Y, Theta)
    J_list[i] = cost(Y_train, Y_pred)
    for t=1:length(Theta)
      Theta[t] = Theta[t] - learning_rate*Theta_grad[t]
    end
  end
  Theta, J_list
end

function predict(X, Theta)
  _, _, y = forward_prop(X, Theta)
  y 
end
