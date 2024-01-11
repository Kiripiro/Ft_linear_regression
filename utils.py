import csv
import os
import numpy as np

def imagesExists():
    if not os.path.isdir('images'):
        os.mkdir('images')

def load_data(file_path):
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File not found at path: {file_path}')
        if not file_path.endswith('.csv'):
            raise Exception(f'File must be a .csv file')

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = data[1:]
            x_data = [float(x) for x, _ in data]
            y_data = [float(y) for _, y in data]
            if len(x_data) != len(y_data):
                raise Exception("x and y data points must be equal in length")
        return x_data, y_data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def load_thetas(file_path):
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File not found at path: {file_path}')

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            theta0 = float(data[0][0])
            theta1 = float(data[0][1])
        return theta0, theta1
    except Exception as e:
        raise Exception(f"loading data: {e}")


def estimated_price(theta0, theta1, x):
    return theta0 + theta1 * x

def normalize(data):
        if not data or len(data) == 1:
            return data

        min_val, max_val = min(data), max(data)
        return [(d - min_val) / (max_val - min_val) for d in data]

def compute_cost(theta0, theta1, x_data, y_data):
    m = len(y_data)
    estimated_prices = theta0 + theta1 * np.array(x_data)
    total_cost = np.sum((estimated_prices - np.array(y_data)) ** 2)
    return total_cost / (2 * m)