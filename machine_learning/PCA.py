class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, X):

        ## 减去均值
        X_scale = X - np.mean(X, axis=0, keepdims=True)

        ## 计算协方差
        X_cov = np.cov(X_scale.T)

        ## 求出特征值和特征矩阵
        _, variance, components  = np.linalg.svd(X_cov)

        ## 对产生的特征值进行排序后选择前k个
        index_feature = np.argsort(variance[::-1])[: self.n_components]

        ## 选择 前k 个最大成分
        self.explained_variance_ = variance[index_feature]
        self.components_ =  -components[index_feature, :]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(variance) 
    
    def fit_transform(self, X):
        self.fit(X)
        X_transform = np.dot(X, self.components_.T)
        
        return X_transform
    
    def transform(self, X):
        X_transform = np.dot(X, self.components_.T)
        
        return X_transform
