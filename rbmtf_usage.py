from rbmtf import RBM
import numpy as np

# this is a 3 hidden, 4 visible node rbm
test = np.array([[0,1,1,0], [0,1,0,0], [0,0,1,1]])
# visible_dim=4 hidden_dim=3 learning_rate=0.1 number_of_iterations=100
rbm = RBM(4, 3, 0.1, 100)
rbm.train(test)



