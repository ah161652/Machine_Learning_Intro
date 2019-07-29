import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def least_squares_linear(x, y):
    xt = np.transpose(np.matrix(x))
    n, m = xt.shape
    x1 = np.ones((n, 1))

    x_matrix = np.hstack((x1, xt))

    y_matrix = np.matrix.transpose(np.matrix(y))

    als = np.linalg.inv((np.matrix.transpose(x_matrix) * (x_matrix))) * (np.matrix.transpose(x_matrix)) * y_matrix
    c = (als[0].item())
    m1 = (als[1].item())
    return c, m1


def least_squares_3(x, y):
    xt = np.transpose(np.matrix(x))
    n, m = xt.shape
    x1 = np.ones((n, 1))
    x2 = np.square(xt)
    x3 = np.power(xt, 3)

    x_matrix = np.hstack((x1, xt, x2, x3))
    y_matrix = np.matrix.transpose(np.matrix(y))

    als = np.linalg.inv((np.matrix.transpose(x_matrix) * (x_matrix))) * (np.matrix.transpose(x_matrix)) * y_matrix
    c = (als[0].item())
    m1 = (als[1].item())
    m2 = (als[2].item())
    m3 = (als[3].item())
    return c, m1, m2, m3


def least_squares_sin(x,y):
    xt = np.transpose(np.matrix(x))
    n, m = xt.shape
    x_sin = np.sin(xt)
    x1 = np.ones((n, 1))

    x_matrix = np.hstack((x1, x_sin))
    y_matrix = np.matrix.transpose(np.matrix(y))

    als = np.linalg.inv((np.matrix.transpose(x_matrix) * (x_matrix))) * (np.matrix.transpose(x_matrix)) * y_matrix
    c = (als[0].item())
    m1 = (als[1].item())
    return c, m1


def linear_error(x_points, y_points, c, m1):
    i = 0
    err = 0
    while i < len(y_points):
        y = y_points[i]
        y_hat = c + (m1 * x_points[i])
        error = (y_hat - y) ** 2
        err = error + err
        i = i + 1
    return err


def cubed_error(x_points, y_points, c, m1, m2, m3):
    i = 0
    err = 0
    while i < len(y_points):
        y = y_points[i]
        y_hat = c + (m1 * x_points[i]) + (m2 * (x_points[i]**2)) + (m3 * (x_points[i]**3))
        error = (y_hat - y) ** 2
        err = error + err
        i = i + 1
    return err


def sin_error(x_points, y_points, c, m1):
    i = 0
    err = 0
    while i < len(y_points):
        y = y_points[i]
        y_hat = c + (m1 * np.sin(x_points[i]))
        error = (y_hat - y) ** 2
        err = error + err
        i = i + 1
    return err


def sin_handler(segments_x, segments_y, num_segments, x):
    intercepts = [None] * num_segments
    gradients = [None] * num_segments

    j = 0
    while j < num_segments:
        intercepts[j], gradients[j] = least_squares_sin(segments_x[j], segments_y[j])
        j = j + 1

    errors = [None] * num_segments

    k = 0
    while k < num_segments:
        errors[k] = sin_error(segments_x[k], segments_y[k], intercepts[k], gradients[k])
        k = k + 1

    equations = [None] * num_segments
    p = 0
    while p < num_segments:
        equations[p] = intercepts[p] + (gradients[p] * np.sin(x))
        p = p + 1

    return equations, errors


def linear_handler(segments_x, segments_y, num_segments, x):
    intercepts = [None] * num_segments
    gradients = [None] * num_segments

    j = 0
    while j < num_segments:
        intercepts[j], gradients[j] = least_squares_linear(segments_x[j], segments_y[j])
        j = j + 1

    errors = [None] * num_segments

    k = 0
    while k < num_segments:
        errors[k] = linear_error(segments_x[k], segments_y[k], intercepts[k], gradients[k])
        k = k + 1

    equations_linear = [None] * num_segments
    p = 0
    while p < num_segments:
        equations_linear[p] = intercepts[p] + (gradients[p] * x)
        p = p + 1

    return equations_linear, errors


def cubed_handler(segments_x, segments_y, num_segments, x):
    intercepts = [None] * num_segments
    m1 = [None] * num_segments
    m2 = [None] * num_segments
    m3 = [None] * num_segments

    t = 0
    while t < num_segments:
        intercepts[t], m1[t], m2[t], m3[t] = least_squares_3(segments_x[t], segments_y[t])
        t = t + 1

    errors = [None] * num_segments

    k = 0
    while k < num_segments:
        errors[k] = cubed_error(segments_x[k], segments_y[k], intercepts[k], m1[k], m2[k], m3[k])
        k = k + 1

    equations = [None] * num_segments
    p = 0
    while p < num_segments:
        equations[p] = intercepts[p] + (m1[p] * x) + (m2[p] * (x**2)) + (m3[p] * (x**3))
        p = p + 1

    return equations, errors


data_x, data_y = load_points_from_file(sys.argv[1])

assert len(data_x) == len(data_y)
assert len(data_x) % 20 == 0
len_data = len(data_x)
num_segments = len_data // 20
segments_x = [None] * num_segments
segments_y = [None] * num_segments
final_errors = [None] * num_segments
colour = np.concatenate([[i] * 20 for i in range(num_segments)])
plt.set_cmap('Dark2')
plt.scatter(data_x, data_y, c=colour)
x = np.linspace(min(data_x), max(data_x), 100)

i = 0
while i < num_segments:
    segments_x[i] = data_x[i*20:(i*20)+20]
    segments_y[i] = data_y[i * 20:(i * 20) + 20]
    i = i + 1

linear_equations, linear_errors = linear_handler(segments_x, segments_y, num_segments, x)
cubed_equations, cubed_errors = cubed_handler(segments_x, segments_y, num_segments, x)
sin_equations, sin_errors = sin_handler(segments_x, segments_y, num_segments, x)

j = 0
while j < num_segments:
    if linear_errors[j] == min(linear_errors[j], cubed_errors[j], sin_errors[j]):
        final_errors[j] = linear_errors[j]
        plt.plot(x, linear_equations[j], '')
        # print('linear')
    elif cubed_errors[j] == min(linear_errors[j], cubed_errors[j], sin_errors[j]):
        if linear_errors[j] - cubed_errors[j] < 0.20 * cubed_errors[j]:
            final_errors[j] = linear_errors[j]
            plt.plot(x, linear_equations[j], '')
            # print('linear')
        elif sin_errors[j] - cubed_errors[j] < 0.20 * cubed_errors[j]:
            final_errors[j] = sin_errors[j]
            plt.plot(x, sin_equations[j], '')
            # print('sin')
        else:
            final_errors[j] = cubed_errors[j]
            plt.plot(x, cubed_equations[j], '')
            # print('cubed')

    elif sin_errors[j] == min(linear_errors[j], cubed_errors[j], sin_errors[j]):
        if linear_errors[j] - sin_errors[j] < 0.20 * sin_errors[j]:
            final_errors[j] = linear_errors[j]
            plt.plot(x, linear_equations[j], '')
            # print('linear')
        else:
            final_errors[j] = sin_errors[j]
            plt.plot(x, sin_equations[j], '')
            # print('sin')
    j = j + 1

try:
    plot_input = sys.argv[2]
except IndexError:
    plot_input = 'null'

if plot_input == '--plot':
    plt.show()

print(sum(final_errors))
  
          
  
  
  
  
      
      
  
