class TSne:
    def __init__(self, data, perplexity = 33, lr = 0.04, n_iterations =1000, momentum =0.4):
        self.data = data
        self.perp = perplexity
        self.lr = lr
        self.n_iterations = n_iterations
        self.mumentum = momentum
        self.n, self.m = self.data.shape
        self.diff = pairwise_distances(self.data, metric='euclidean') + 10**(-8)
        
    def perplexity_calc(self, p):
        return np.power(2, -np.sum(p * np.log2(p)))
        
    def p_cal(self, diff, sigmas):
        s = np.ones(self.n)
        for i in range(self.n):
            s[i] = np.exp(-diff[i] ** 2 / (2 * sigmas[i]))
        summ = np.sum(s)
        pi = [s[i]/summ for i in range(self.n)]
        return pi
    
    def binary_search(self, point):
        
        max_iterations = 47
        tol = 10**(-4)
        sigma = np.random.randn(self.n)
        sigma_min = np.array([0 for i in range(self.n)])
        sigma_max = np.array([10**6 for i in range(self.n)])
        
        for i in range(max_iterations):
        
            p = self.p_cal(self.diff[point], sigma)
            perplexity = self.perplexity_calc(p)
            
            if np.abs(perplexity - self.perp) < tol:
                break


            if perplexity > self.perp:
                sigma_max = (sigma_min + sigma_max) / 2

            else:
                sigma_min = (sigma_min + sigma_max) / 2
            
            sigma = np.sqrt(sigma_min * sigma_max)


        return sigma
    
#The code is incomplete
