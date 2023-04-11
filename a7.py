from neural import *

training_data = [([0,0], [0]), ([1,0], [1]), ([0,1], [1]), ([1,1], [0])]
print('egg')
xorn = NeuralNet(2,2,1)
xorn.train(training_data)
print(xorn.test_with_expected(training_data))