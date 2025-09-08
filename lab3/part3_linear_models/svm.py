import numpy as np

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge with any method
    # that support constrain.
    # begin answer
    y_svm = np.where(y.T == 0, -1, 1)
    X_T = X.T

    # 定义目标函数
    Q = y_svm @ y_svm.T * (X_T @ X)
    objective = lambda alpha: 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
    
    # 定义等式约束 sum(alpha_i * y_i) = 0
    cons = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_svm.flatten())})
    
    # 定义边界约束 0 <= alpha_i <= C
    C = 1.0
    bounds = tuple([(0, C)] * N)
    
    # 运行优化器
    alpha_opt = minimize(objective, np.zeros(N), method='SLSQP', bounds=bounds, constraints=cons).x
    
    # 从alpha计算w
    w_vec = np.sum(alpha_opt[:, np.newaxis] * y_svm * X_T, axis=0)
    
    # 找到支持向量并计算数量
    sv_indices = alpha_opt > 1e-5
    num = np.sum(sv_indices)
    
    # 计算偏置b
    b = np.mean(y_svm[sv_indices] - np.dot(X_T[sv_indices], w_vec))
    
    # 组合w和b
    w = np.vstack([b, w_vec.reshape(-1, 1)])
    # end answer
    return w, num

