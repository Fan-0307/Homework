import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    X_b = np.vstack([np.ones((1, N)), X])
    w = np.zeros((P + 1, 1))
    learning_rate = 0.01
    num_iterations = 1000
    y = y.reshape((N, 1))
    for i in range(num_iterations):
        # 计算假设函数（Sigmoid 函数）
        # w.T 和 X_b 的点积得到一个 1-by-N 的行向量
        h = 1 / (1 + np.exp(-(w.T @ X_b)))
        
        # 计算误差
        error = h.T - y
        
        # 计算梯度
        # 梯度是 1/N * X_b @ error
        gradient = (1/N) * (X_b @ error)
        
        # 更新权重
        w -= learning_rate * gradient
    # end answer

    
    return w
