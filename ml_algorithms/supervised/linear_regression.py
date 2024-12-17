import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    A simple linear regression model using gradient descent optimization.

    This class implements a linear regression model that tries to find the linear relationship 
    between a set of predictors (features) and a continuous target variable by minimizing 
    the mean squared error (MSE) cost function. The model uses gradient descent to update 
    parameters (weights and bias).

    Attributes
    ----------
    n_iterations : int
        The number of gradient descent iterations to run during training.
    learning_rate : float
        The step size used for each update in the gradient descent optimization.
    weights : np.ndarray or None
        Model weights (coefficients) after training. Initialized as None before fitting.
    bias : float or np.ndarray or None
        Model bias (intercept) after training. Initialized as None before fitting.
    cost_history : list of float
        A history of the cost values for each iteration, useful for analysis and debugging.
    """

    def __init__(self, n_iterations=1000, learning_rate=0.01):
        """
        Initialize the LinearRegression model.

        Parameters
        ----------
        n_iterations : int, optional
            Number of iterations for the gradient descent algorithm. 
            Default is 1000.
        learning_rate : float, optional
            The learning rate or step size for parameter updates.
            Default is 0.01.
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _initial_parameters(self, m, n_features):
        """
        Initialize model parameters (weights and bias).

        This private method sets up the initial weights and bias as zeros. 
        Called internally by the `fit` method.

        Parameters
        ----------
        m : int
            Number of training samples.
        n_features : int
            Number of features in the predictors matrix.
        """
        self.weights = np.zeros((n_features, 1))
        self.bias = np.zeros((1, ))
    
    def _compute_cost(self, predictors, target, m):
        """
        Compute the cost (MSE) for the given set of predictions and targets.

        This private method calculates the mean squared error cost function:
        cost = (1/(2*m)) * Σ(predicted - target)^2

        Parameters
        ----------
        predictors : np.ndarray of shape (m, n_features)
            The input data on which predictions are to be made.
        target : np.ndarray of shape (m, 1)
            The true target values corresponding to the input data.
        m : int
            Number of training samples.

        Returns
        -------
        float
            The computed mean squared error cost.
        """
        predicts = self.predict(predictors)
        cost = (1 / (2 * m)) * np.sum((predicts - target) ** 2)
        return cost
     
    def _compute_gradient(self, predictors, target, m):
        """
        Compute the gradients of the cost function with respect to weights and bias.

        This private method computes partial derivatives of the cost function 
        with respect to model parameters. It returns the gradients for weights and bias 
        which are then used by the `fit` method to update model parameters.

        Parameters
        ----------
        predictors : np.ndarray of shape (m, n_features)
            The input data for gradient computation.
        target : np.ndarray of shape (m, 1)
            The true target values.
        m : int
            Number of training samples.

        Returns
        -------
        dw : np.ndarray of shape (n_features, 1)
            The gradient of the cost with respect to the weights.
        db : float
            The gradient of the cost with respect to the bias.
        """
        predicts = self.predict(predictors)
        dw = (1 / m) * np.dot(predictors.T, (predicts - target))
        db = (1 / m) * np.sum(predicts - target)
        return dw, db
    
    def fit(self, predictors, target):
        """
        Train the linear regression model using gradient descent.

        This method initializes the parameters and runs the gradient descent algorithm 
        for the specified number of iterations (n_iterations). The weights and bias are 
        updated in each iteration. The cost at each iteration is recorded in cost_history.

        Parameters
        ----------
        predictors : np.ndarray of shape (m, n_features)
            The input training data. Each row represents a sample and each column represents a feature.
        target : np.ndarray of shape (m, 1)
            The target values corresponding to the training data.
        """
        m, n_features = predictors.shape
        self._initial_parameters(m, n_features)
        
        for i in range(self.n_iterations):
            dw, db = self._compute_gradient(predictors, target, m)
            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate
            
            cost = self._compute_cost(predictors, target, m)
            self.cost_history.append(cost)
    
    def predict(self, predictors):
        """
        Predict target values using the trained linear regression model.

        Parameters
        ----------
        predictors : np.ndarray of shape (m, n_features)
            Input data for prediction. Each row is a sample, and each column is a feature.

        Returns
        -------
        np.ndarray of shape (m, 1)
            Predicted target values.
        """
        predicts = np.dot(predictors, self.weights) + self.bias
        return predicts
    
    def plot_cost_history(self):
        """
        Plot the cost function values over the training iterations.

        This method provides a visual representation of the cost function’s decrease 
        as the model trains, which can help in diagnosing whether the model converged 
        properly or needs more iterations.

        No parameters, no return value. Displays a matplotlib plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function Convergence')
        plt.grid(True)
        plt.show()