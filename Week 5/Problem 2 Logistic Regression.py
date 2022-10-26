import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    x_data = np.array(df.iloc[:, :-1])  # convert to np array will automatically remove header
    y_data = np.array(df.iloc[:, -1])
    return x_data, y_data


def add_intercept(x_data):
    intercept = np.ones(x_data.shape[0])
    return np.c_[intercept, x_data]


class LogisticRegression:
    def __init__(self):
        self.theta = None
        self.x = None
        self.y = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x.dot(self.theta)))

    def fit(self, x, y, learning_rate, eps):
        self.x = add_intercept(x)
        self.y = y
        m, n = self.x.shape
        self.theta = np.zeros(n)
        while True:
            last_theta = np.copy(self.theta)
            gradient = 1 / m * self.x.T.dot(self.sigmoid(self.x) - self.y)
            self.theta -= learning_rate * gradient
            if np.linalg.norm(self.theta - last_theta) <= eps:
                return self.theta

    def predict(self, x):
        sigmoid = self.sigmoid(add_intercept(x))
        if sigmoid <= 0.5:
            return 0
        else:
            return 1

    def plot(self, save_path):
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        min_x1 = np.min(self.x[:, 1])
        min_x2 = np.min(self.x[:, 2])
        max_x1 = np.max(self.x[:, 1])
        max_x2 = np.max(self.x[:, 2])
        margin1 = (max_x1 - min_x1) * 0.2
        margin2 = (max_x2 - min_x2) * 0.2

        # Plot the x dataset
        plt.plot(self.x[self.y == 0, 1], self.x[self.y == 0, 2], 'co', linewidth=2)
        plt.plot(self.x[self.y == 1, 1], self.x[self.y == 1, 2], 'm^', linewidth=2)

        # Plot the decision boundary
        x = np.arange(min_x1 - margin1, max_x1 + margin1, 0.01)
        y = -(self.theta[0] / self.theta[2] + self.theta[1] / self.theta[2] * x)
        plt.plot(x, y, color='red', linewidth=2)
        plt.xlim(min_x1 - margin1, max_x1 + margin1)
        plt.ylim(min_x2 - margin2, max_x2 + margin2)
        plt.xlabel('Salary')
        plt.ylabel('Experience')
        plt.savefig(save_path)
        plt.show()


def main():
    x, y = load_dataset('dataset.csv')
    model = LogisticRegression()
    theta = model.fit(x, y, learning_rate=0.1, eps=0.0001)
    print(f'Estimated coefficient for logistic regression model is\n theta = {theta}')
    model.plot(save_path='output.png')


if __name__ == "__main__":
    main()

