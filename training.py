import os
import argparse
import logging
import numpy as np
import imageio as io
import matplotlib.pyplot as plt

from utils import *

blue_code = '\033[94m'
orange_code = '\033[93m'
end_code = '\033[0m'

class LinearRegression():
    def __init__(self, fileName='data.csv', theta0=0, theta1=0, learning_rate=0.1, max_iterations=1000, epsilon=1e-6, gif=False, showPlots=False):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.x_data, self.y_data = load_data(fileName)
        self.x_data_normalized = normalize(self.x_data)
        self.y_data_normalized = normalize(self.y_data)
        self.theta0 = theta0
        self.theta1 = theta1
        self.raw_theta0 = 0
        self.raw_theta1 = 0
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.precision = 0
        self.cost = 0
        self.iterations = 0
        self.gif = gif
        self.previous_cost = float('inf')
        self.epsilon = epsilon
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.errors = []
        self.show = showPlots

    def denormalize_thetas(self, theta0, theta1):
        x_data = np.array(self.x_data)
        y_data = np.array(self.y_data)
        min_x, max_x = min(x_data), max(x_data)
        min_y, max_y = min(y_data), max(y_data)
        theta0 = min_y + theta0 * (max_y - min_y) - theta1 * min_x * (max_y - min_y) / (max_x - min_x)
        theta1 = (theta1 * (max_y - min_y)) / (max_x - min_x)
        return theta0, theta1

    def makeGif(self):
        images = []
        for i in range(0, self.iterations, 10):
            images.append(io.v2.imread(f"images/iteration_{i}.png"))
        io.mimsave('images/animation.gif', images, fps=10)
        logging.info(f"Gif saved to {orange_code}./images/animation.gif{end_code}")

    def fit(self):
        m = len(self.y_data_normalized)

        for self.iterations in range(self.max_iterations):
            theta0_gradient = 0
            theta1_gradient = 0
            for x, y in zip(self.x_data_normalized, self.y_data_normalized):
                theta0_gradient += estimated_price(self.theta0, self.theta1, x) - y
                theta1_gradient += (estimated_price(self.theta0, self.theta1, x) - y) * x
            self.theta0 -= (self.learning_rate / m) * theta0_gradient
            self.theta1 -= (self.learning_rate / m) * theta1_gradient

            current_cost = compute_cost(self.theta0, self.theta1, self.x_data_normalized, self.y_data_normalized)
            self.errors.append(current_cost)
            self.plot()
            if self.gif and (self.iterations % 10 == 0 or self.iterations == 0):
                plt.savefig(f"images/iteration_{self.iterations}.png")
            if abs(current_cost - self.previous_cost) < self.epsilon:
                logging.info(f"{blue_code}Converged after {orange_code}{self.iterations}{end_code} {blue_code}iterations{end_code}")
                break
            self.previous_cost = current_cost
        if self.gif:
            self.makeGif()
        self.precision = self.get_precision(self.x_data_normalized, self.y_data_normalized)
        self.cost = compute_cost(self.theta0, self.theta1, self.x_data_normalized, self.y_data_normalized)
        self.plot(final=True)

    def predict(self, x_data):
        return estimated_price(self.theta0, self.theta1, x_data)

    def score(self, x_data, y_data):
        return compute_cost(self.theta0, self.theta1, x_data, y_data)

    def get_precision(self, x_data, y_data):
        mse = compute_cost(self.theta0, self.theta1, x_data, y_data)
        precision = 1 / mse
        rounded_precision = round(precision, 5)
        return rounded_precision

    def plot(self, final=False):
        self.ax1.clear()
        self.ax1.scatter(self.x_data_normalized, self.y_data_normalized)
        self.ax1.plot(self.x_data_normalized, [self.predict(x) for x in self.x_data_normalized], color='red')
        self.ax1.set_title('Normalized Linear Regression')
        self.ax1.set_xlabel('Normalized mileage')
        self.ax1.set_ylabel('Normalized price')

        self.ax2.plot(self.errors, color='blue')
        self.ax2.set_title("Cost Curve")
        self.ax2.set_xlabel("Iterations")
        self.ax2.set_ylabel("Cost")
        if self.show and final:
            plt.show()
            for f in os.listdir('images'):
                if f.endswith('.png'):
                    os.remove(os.path.join('images', f))

    def plot_reverse_normalized(self, theta0, theta1):
        plt.figure()
        plt.scatter(self.x_data, self.y_data)
        plt.plot(self.x_data, [theta0 + theta1 * x for x in self.x_data], color='red')
        plt.suptitle(f'Linear Regression')
        plt.annotate(f"LR: {self.learning_rate}\nEpsilon: {self.epsilon},\nIter={self.iterations}\nTheta0: {round(theta0, 5)}\nTheta1: {round(theta1, 5)}\nPrecision: {self.precision}\nCost: {round(self.cost, 5)}", 
            xy=(0.80, 0.80), 
            xycoords='axes fraction', 
            bbox=dict(boxstyle="round", 
            fc="w"))
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.savefig(f"images/linear_regression.png")
        logging.info(f"Plot using raw data saved to {orange_code}./images/linear_regression.png{end_code}\n")
        if self.show:
            plt.show()
        for f in os.listdir('images'):
            if f.endswith('.png') and not f.startswith('linear_regression'):
                os.remove(os.path.join('images', f))

    def save_N_print_raw_thetas(self):
        self.raw_theta0, self.raw_theta1 = self.denormalize_thetas(self.theta0, self.theta1)
        with open('raw_thetas.csv', 'w') as f:
            f.write(f"{self.raw_theta0},{self.raw_theta1}")
            logging.info(f"Raw theta0: {orange_code}{self.raw_theta0}{end_code}, raw theta1: {orange_code}{self.raw_theta1}{end_code}, saved to {orange_code}./raw_thetas.txt{end_code}")

        self.plot_reverse_normalized(self.raw_theta0, self.raw_theta1)

    def output(self):
        result = f"Base input file: {orange_code}data.csv{end_code}\n"
        result += f"Base values:\n"
        result += f"    Learning rate: {orange_code}{self.learning_rate}{end_code}\n"
        result += f"    Max iterations: {orange_code}{self.max_iterations}{end_code}\n"
        result += f"    Epsilon: {orange_code}{self.epsilon}{end_code}"
        result += "\n\n"
        result += f"Normalized values:\n"
        result += f"    Normalized theta0: {orange_code}{self.theta0}{end_code}\n"
        result += f"    Normalized theta1: {orange_code}{self.theta1}{end_code}\n"
        result += f"    Precision: {orange_code}{self.precision}{end_code}\n"
        result += f"    Cost: {orange_code}{self.cost}{end_code}\n"
        result += f"Iterations: {orange_code}{self.iterations}{end_code}"
        return result

    def main(self):
        imagesExists()
        logging.info(f"{blue_code}Starting linear regression training...{end_code}")
        if self.gif:
            logging.info(f"Gif will be saved to {orange_code}./images/animation.gif{end_code}")
        self.fit()
        self.save_N_print_raw_thetas()
        logging.info(self.output())

    def prompt(self):
        print(f"{blue_code}Welcome to the Linear Regression Car Price Estimator!\n{end_code}")
        if os.path.exists('raw_thetas.csv'):
            user_input = input(f"{orange_code}Raw thetas file already exists. Do you want to run the program again? (y/n):{end_code} ")
            if user_input.lower() == 'y':
                self.main()
            else:
                logging.info("Exiting the program.")
                exit()
        else:
            self.main()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Regression with Command-Line Arguments')
    parser.add_argument('--f', type=str, default='data.csv', help='Input data file as .csv')
    parser.add_argument('--t0', type=float, default=0, help='Initial value for theta0')
    parser.add_argument('--t1', type=float, default=0, help='Initial value for theta1')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--mi', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--ep', type=float, default=1e-6, help='Convergence threshold for epsilon')
    parser.add_argument('--gif', action='store_true', help='Generate GIF of iterations')
    parser.add_argument('--show', action='store_true', help='Show plots')
    
    args = parser.parse_args()

    lr = LinearRegression(
        fileName=args.f,
        theta0=args.t0,
        theta1=args.t1,
        learning_rate=args.lr,
        max_iterations=args.mi,
        epsilon=args.ep,
        gif=args.gif,
        showPlots=args.show
    )
    lr.prompt()