import numpy as np
import matplotlib.pyplot as plt


def add_intercept(x):
    new_x = np.c_[np.ones(x.shape[0]), x]
    return new_x


def load_dataset(file_path):
    x_data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=0)
    y_data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=1)
    return x_data, y_data


def plot(x, y, theta, save_path):
    # Plot the dataset
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y, 'co', linewidth=2)

    # Plot the prediction line
    margin1 = (np.max(x) - np.min(x)) * 0.2
    margin2 = (np.max(y) - np.min(y)) * 0.2
    x_pred = np.arange(np.min(x), np.max(x))
    y_pred = theta[0] + theta[1] * x_pred
    plt.plot(x_pred, y_pred, color='red', linewidth=2)

    plt.xlim(np.min(x) - margin1, np.max(x) + margin1)
    plt.ylim(np.min(y) - margin2, np.max(y) + margin2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_path)
    plt.show()


class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, x, y):
        x = add_intercept(x)
        p = np.linalg.inv(x.T.dot(x))
        self.theta = p.dot(x.T).dot(y)
        return self.theta

    def predict(self, inp):
        return self.theta[0] + inp * self.theta[1]


def main():
    x, y = load_dataset(file_path=r"D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week "
                                  r"3\data\data_linear.csv")
    model_1 = LinearRegression()
    theta = model_1.fit(x, y)
    print(f'Estimated coefficient for linear model is\n w0 = {theta[0]} \nw1 = {theta[1]}')
    plot(x, y, theta, save_path="D:\DSEB LEARNINGB MATERIAL\ML 1\Machine-Learning-1-DSEB62\Week 3\output "
                                "image\Problem2.png")
    for i in [50, 100, 150]:
        print(f"Predicted price for house with area {i} is {model_1.predict(i)}")


if __name__ == "__main__":
    main()
