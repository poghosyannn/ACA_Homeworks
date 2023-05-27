class DenseLayer:
    def __init__(self, size):
        self.size = size
    
    def initialize(self, neur_prev):

        
class NN:
    def __init__(self, layers_list, lr = 0.03, max_iter = 100):
        self.ll = layers_list
        self.lr = lr
        self.max_iter = max_iter
        
    def construction(self):
        sizes = [i.size for i in self.ll]
        self.weights_nn = []
        for i in range(1, len(sizes)):
            self.ll[i].initialize(sizes[i-1])
            self.weights_nn.append(self.ll[i].w)
        
                  
    def feed_forward(self, input):
        prod = input
        self.values = []
        self.values.append(prod)

        for i in self.weights_nn:
            prod = i @ np.insert(prod, 0, 1)
            self.values.append(prod)

        return prod
    
    def back_propagation(self, output, y):

#         g = (-2 * self.values[-1].T @ (y - (output))) / self.values[-1].shape[0]
        g = (-2 * (y - (output))) / self.values[-1].shape[0]
    
        # then we just get the Jacobians for the weights by chain rule. I do not manage to complete this part because,
        # unexpectedly, I had to submit this hw one day earlier. 
        
        return g
        
                    
        
l = [DenseLayer(4), DenseLayer(3), DenseLayer(2)]
nn = NN(l)
nn.construction()

a = nn.feed_forward(np.array([1, 15, 11, 40]))
print(nn.values)


nn.back_propagation(a, np.array([11, 10]))

