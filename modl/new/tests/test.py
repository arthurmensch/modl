import numpy as np
from modl.new.dict_fact_new import DictFactNew

dict_init = np.random.randn(100, 200)
dl = DictFactNew(dict_init, n_samples=200)
X = np.random.randn(100, 200)

dl.partial_fit(X, np.arange(200))
