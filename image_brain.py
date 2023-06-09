from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, "densenet121-a639ec97.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "ft_net.jpeg"))
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, ":", eachProbability)
