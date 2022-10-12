import numpy as np
import matplotlib.pyplot as plt
import random


def create_data(n):
    """Create dataset with length = n"""
    x_data = np.arange(0, 1.1, 1 / n)
    noise = np.arange(-0.2, 0.3, 0.1)
    y_data = np.array([])
    for e in x_data:
        err = random.choice(noise)
        y_data = np.append(y_data, np.sin(2 * np.pi * e) + err)
    return x_data, y_data


def insert_column(x_data, p):
    """Add more data for polynomial order p and intercept"""
    if p == 0:  # p is the polynomial order
        return x_data
    intercept = np.ones(x_data.shape[0])
    new_x = np.copy(x_data)
    for i in range(2, p + 1):
        new_x = np.column_stack((new_x, x_data ** i))
    new_x = np.c_[intercept, new_x]  # always use column stack for these cases
    return new_x


class PolynomialLinearRegression:
    def __init__(self, x, y, order):
        self.x = x
        self.y = y
        self.order = order
        self.theta = None

    def normal_fit(self):
        if self.order == 0:
            self.theta = np.zeros(self.x.shape[0])
            return self.theta
        x_new = insert_column(self.x, self.order)
        p = np.linalg.inv(x_new.T.dot(x_new))
        self.theta = p.dot(x_new.T).dot(self.y)
        return self.theta

    def ridge_fit(self, alpha):
        if self.order == 0:
            self.theta = np.zeros(self.x.shape[0])
            return self.theta
        x_new = insert_column(self.x, self.order)
        p = x_new.T.dot(x_new) + alpha * x_new.shape[0] * np.identity(x_new.shape[1])
        self.theta = np.linalg.inv(p).dot(x_new.T).dot(self.y)
        return self.theta

    def lasso_fit(self, alpha, eps, learning_rate):
        n = self.x.shape[0]
        if self.order == 0:
            return self.theta
        x_new = insert_column(self.x, self.order)
        self.theta = np.zeros(x_new.shape[1])
        while True:
            g1 = 2 * x_new.T.dot(self.y - x_new.dot(self.theta))
            g2 = alpha * np.sign(self.theta)
            gradient = 1 / n * g1 + g2
            self.theta += learning_rate * gradient
            print(np.linalg.norm(gradient))
            if np.linalg.norm(gradient) <= eps:
                return self.theta

    def predict(self, inp):
        if self.order == 0:  # if order = 0, return mean of dependent variables
            return np.array([np.mean(self.y)] * len(inp))
        inp = insert_column(inp, self.order)
        return inp.dot(self.theta)

    def plot(self, save_path):
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.plot(self.x, self.y, 'co', linewidth=2)
        min_x = np.min(self.x)
        min_y = np.min(self.y)
        max_x = np.max(self.x)
        max_y = np.max(self.y)
        margin1 = (max_x - min_x) * 0.2
        margin2 = (max_y - min_y) * 0.2
        x_line = np.arange(min_x, max_x, 0.001)
        y_predict = self.predict(x_line)
        y_real = np.sin(2 * np.pi * x_line)
        plt.plot(x_line, y_predict, color='red', linewidth=2, label='Prediction')
        plt.plot(x_line, y_real, color='green', linewidth=2, label='sin(2 pi x)')
        plt.legend(loc='upper right')
        plt.xlim(min_x - margin1, max_x + margin1)
        plt.ylim(min_y - margin2, max_x + margin2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(save_path)
        plt.show()


def main():
    # Initial data generation
    x, y = create_data(10)
    # Part a: fit polynomial order = 0, 3, 6, 9 and plot
    for order in [0, 3, 6, 9]:
        model = PolynomialLinearRegression(x, y, order)
        theta = model.normal_fit()
        print(f'Theta for Polynomial Regression model with order 0 is {theta}')
        model.plot(save_path=f"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 4\output image\Problem "
                             f"2 part a model order {order}.png")

    # Part b: fit polynomial order = 9 with larger dataset
    x1, y1 = create_data(25)
    x2, y2 = create_data(110)
    model1 = PolynomialLinearRegression(x1, y1, order=9)
    theta1 = model1.normal_fit()
    print(f'Theta for Polynomial Regression model with order 9, dataset n = 25 is {theta1}')
    model2 = PolynomialLinearRegression(x2, y2, order=9)
    theta2 = model2.normal_fit()
    print(f'Theta for Polynomial Regression model with order 9, dataset n = 110 is {theta2}')
    model1.plot(save_path=f"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 4\output image\Problem 2 "
                          f"part b small.png")
    model2.plot(save_path=f"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 4\output image\Problem 2 "
                          f"part b large.png")

    # Part c: fit ridge regression and lasso regression for initial dataset, poly order = 9
    model = PolynomialLinearRegression(x, y, order=9)
    theta_ridge = model.ridge_fit(alpha=0.0001)
    print(f"Theta for Ridge Regression poly order=9 is {theta_ridge}")
    model.plot(
        save_path=f"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 4\output image\Problem 2 Ridge Regression.png")
    theta_lasso = model.lasso_fit(alpha=0.045, learning_rate=0.1, eps=0.1)
    print(f"Theta for Lasso Regression poly order=9 is {theta_lasso}")
    model.plot(save_path=f"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 4\output image\Problem 2 Lasso Regression.png")


if __name__ == "__main__":
    main()