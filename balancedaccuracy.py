import torch

# NOTE: This will be the calculation of balanced accuracy for your classification task
# The balanced accuracy is defined as the average accuracy for each class.
# The accuracy for an indiviual class is the ratio between correctly classified example to all examples of that class.
# The code in train.py will instantiate one instance of this class.
# It will call the reset methos at the beginning of each epoch. Use this to reset your
# internal states. The update method will be called multiple times during an epoch, once for each batch of the training.
# You will receive the network predictions, a Tensor of Size (BATCHSIZExCLASSES) containing the logits (output without Softmax).
# You will also receive the groundtruth, an integer (long) Tensor with the respective class index per example.
# For each class, count how many examples were correctly classified and how many total examples exist.
# Then, in the getBACC method, calculate the balanced accuracy by first calculating each individual accuracy
# and then taking the average.


# Balanced Accuracy
class BalancedAccuracy:
    def __init__(self, nClasses):
        # TODO: Setup internal variables
        # NOTE: It is good practive to all reset() from here to make sure everything is properly initialized
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.correct = [0] * self.nClasses
        self.total = [0] * self.nClasses

    def update(self, predictions, groundtruth):
        _, predicted = torch.max(predictions, 1)
        for i in range(len(groundtruth)):
            self.total[groundtruth[i].item()] += 1
            if predicted[i] == groundtruth[i]:
                self.correct[groundtruth[i].item()] += 1

    def getBACC(self):
        accuracies = [
            self.correct[i] / self.total[i] if self.total[i] > 0 else 0
            for i in range(self.nClasses)
        ]
        return sum(accuracies) / len(accuracies)  # based on current internal state
