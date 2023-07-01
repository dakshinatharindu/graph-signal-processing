import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# print(adjacency_matrix.shape)

def volterra_filter(signal, adjacency_matrix, num_taps, coefficients):

    c_0 = coefficients[0]
    c_1 = coefficients[1:num_taps]
    c_2 = np.reshape(coefficients[num_taps+1:],(num_taps,num_taps))

    y = c_0 * np.ones(signal.shape[0])

    for k in range(num_taps):
        y += c_1[k] * np.power(adjacency_matrix, k) @ signal

    for k_1 in range(num_taps):
        for k_2 in range(num_taps):
            y += c_2[k_1, k_2] * np.multiply(np.power(adjacency_matrix, k_1) @ signal,  np.power(adjacency_matrix, k_2) @ signal)


    return y


def error_function(x_sample, A_strong, num_taps, coefficient, mask):

    return np.sum(np.square(x_sample - np.multiply(volterra_filter(x_sample, A_strong, num_taps, coefficient), mask)))


def grad(x_sample, c_pred, A_strong, num_taps, mask):

    dx = 0.001
    grad = np.zeros(c_pred.shape[0])
    cost = np.sum(np.square(x_sample - np.multiply(volterra_filter(x_sample, A_strong, num_taps, c_pred), mask)))

    for k in range(1, len(c_pred)):
        dH = np.zeros(c_pred.shape[0])
        dH[k] = dx
        c_plus = c_pred + dH
        c_minus = c_pred - dH
        grad[k] = np.sum()

    
    return (error_function(x_sample, A_strong, num_taps, c_plus, mask) - error_function(x_sample, A_strong, num_taps, c_minus, mask))/(2*dx), cost


def main(x, step_count = 5,  prob=0.5):
    timeseries = np.roll(np.identity(step_count), 1, axis=1)
    timeseries[-1,0] = 0

    A_strong = np.kron(adjacency_matrix, timeseries) + np.kron(adjacency_matrix, np.identity(step_count)) + np.kron(np.identity(adjacency_matrix.shape[0]), timeseries)

    N = 10
    num_taps = 2
    c_pred = np.random.rand(7)
    lr = 3e-5
    costs = np.zeros(N)
    test_error = np.zeros(N)

    mask = np.ones(x.shape[0])[np.random(x.shape[0]) > prob]
    x_sample =  np.multiply(x, mask)
 


    for k in range(N):
        grad, cost = grad(x_sample, c_pred, A_strong, num_taps, mask)
        c_pred =  c_pred - lr * grad
        costs[k] = cost
        x_pred = volterra_filter(x_sample, A_strong, num_taps, c_pred)
        test_error[k] = np.sum(np.square(x - x_pred))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(costs) 
    ax[1].plot(test_error)

    plt.show()


if __name__ == "__main__":
    adjacency_matrix = pd.read_csv("D:/OneDrive - University of Moratuwa/A-Research/volterra_experiments/graph-signal-processing/volterra-experiments/matlab_code/adjacency.csv",header=None)
    x = pd.read_csv("D:/OneDrive - University of Moratuwa/A-Research/volterra_experiments/graph-signal-processing/volterra-experiments/matlab_code/covid_global.csv")
    x = x.iloc[:, 4:9]
    x = x.T 
    x = x.flatten()

    x_max = max(x)
    x = x/x_max
    

    main(x)




    




