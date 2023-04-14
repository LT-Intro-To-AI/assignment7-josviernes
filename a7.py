from neural import *

training_data = [([0,0], [0]), ([1,0], [1]), ([0,1], [1]), ([1,1], [0])]

xorn = NeuralNet(2,1,1)
xorn.train(training_data)
print(xorn.test_with_expected(training_data))

print()
print("\n\nTraining Voter Opinion\n\n")
print()

voter_training_data = [
    ([0.9,0.6,0.8,0.3,0.1],[1.0]),
    ([0.8,0.8,0.4,0.6,0.4],[1.0]),
    ([0.7,0.2,0.4,0.6,0.3],[1.0]),
    ([0.5,0.5,0.8,0.4,0.8],[0.0]),
    ([0.3,0.1,0.6,0.8,0.8],[0.0]),
    ([0.6,0.3,0.4,0.3,0.6],[0.0])
]

voter = NeuralNet(5,30,1)
voter.train(voter_training_data)
print("\nVoter Training Data\n")
print(voter.test_with_expected(voter_training_data))
print()

print("\nVoter Testing Data\n")
print()
voter_testing_data = [
    ([1.0,1.0,1.0,0.1,0.1]),
    ([0.5,0.2,0.1,0.7,0.7]),
    ([0.8,0.3,0.3,0.3,0.8]),
    ([0.8,0.3,0.3,0.8,0.3]),
    ([0.9,0.8,0.8,0.3,0.6])
]

print(f"{voter.test(voter_testing_data)}")