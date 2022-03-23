import configs
import numpy as np

test = np.array([[[1, 0, 0], [0, 1, 0], [0, 1, 0]]])
# print(test)
test_pop = configs.BosonConfig(test)

#print(test_pop.is_legal())
# print(test_pop.shrink(0))
print(test_pop.add_boson(0, [7, 7]))

# print(test_pop.identifier())
