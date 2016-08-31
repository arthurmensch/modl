import numpy as np

from modl.new.dict_fact_new import DictFact

dl = DictFact(n_samples=200, solver='gram', random_state=0, n_epochs=2,
              pen_l1_ratio=0.9, reduction=2)
X = np.random.randn(100, 200)

dl.fit(X)
print(dl.G)
