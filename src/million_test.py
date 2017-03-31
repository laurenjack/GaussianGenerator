import numpy as np
z_dict = {(i, 1) : np.random.randn(5,1) for i in xrange(1000000)}
print "Dict created"

print z_dict[(456901, 1)]
print z_dict[(999998, 1)]