class KMeans():
    from numpy.linalg import norm
    
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        
    
    def fit(self, X):
        ## 随机选取 k 个点作为簇心
        self.cluster_centers_ = X[np.random.randint(0, len(X), size=self.n_clusters), :]
        
        dis = np.zeros([len(X), self.n_clusters])
        ## 迭代    
        for j in range(self.max_iter):
            ## 计算各点到选取点的距离
            for i in range(self.n_clusters):
                dis[:, i] = norm(X - self.cluster_centers_[i, :], axis=1)

            ## 根据距离重新分配各点簇类
            self.labels_ = np.argmin(dis, axis=1)

            ## 选取新簇的中心
            cluster_centers_new = np.zeros_like(self.cluster_centers_)
            for i in np.unique(self.labels_):
                cluster_centers_new[i, :] = np.mean(X[self.labels_ == i, :], axis=0)

            ## 判断中心是否再发生变化
            if norm(self.cluster_centers_ - cluster_centers_new) < self.tol:
                break
            else:
                self.cluster_centers_ = cluster_centers_new
            
        self.inertia_ = norm(dis[self.labels_])
        
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        dis = np.zeros([len(X), self.n_clusters])
        for i in range(self.n_clusters):
            dis[:, i] = norm(X - self.cluster_centers_[i, :], axis=1)
        labels_ = np.argmin(dis, axis=1)
        return labels_