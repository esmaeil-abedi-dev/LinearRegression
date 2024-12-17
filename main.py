import numpy as np
import matplotlib.pyplot as plt
from ml_algorithms.supervised import LinearRegression

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    X = 2 * np.random.rand(20, 5)  # 20 samples, 5 features
    w = np.array([3, 2, -1, 0.5, 1])  # coefficients for each of the 5 features
    y = 4 + X.dot(w.reshape(-1,1)) + np.random.randn(20,1)  # shape is (20,1)
    # np.random.seed(0)
    # X = 2 * np.random.rand(20, 1)  # shape is (20,1)
    # y = 4 + 3 * X + np.random.randn(20, 1)  # shape is (20,1)
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X, y, color='blue', label='Data points')
    # plt.plot(X, y_pred, color='red', label='Linear regression')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Linear Regression Example')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # Plot cost history
    model.plot_cost_history()