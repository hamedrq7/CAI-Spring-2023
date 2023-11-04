## Multi-Domain Classification - Phase 2

In this project, you will work on multi-domain classification using the Random Forest classifier. The dataset is composed of five different domains, each having classes for numbers 0 to 9. Your task involves two main steps:

![Intro](int.png)
#### Part 1 - Digit Classification

In the first part, you will directly classify the numbers from the dataset using a Random Forest algorithm. Your challenge is to choose appropriate hyperparameters to achieve the desired accuracy. You need to demonstrate that your hyperparameter choices are effective. Additionally, you should calculate the confusion matrix to assess your model's performance.
#### Part 2 - Domain and Digit Classification

In the second part, you must first use a Random Forest algorithm to classify the domains of the data effectively. Then, you should create another Random Forest classifier for each domain, specifically for digit classification. To evaluate your model's accuracy, you will need to pass the test data through the domain classifier and use the result to classify the digit with a domain-based digit classifier. Once again, selecting the right hyperparameters is crucial, and you should prove their effectiveness.

#### Part 3 - Using Data from Other Domains in Digit Classification

In real-world scenarios, data might contain noise or be influenced by other factors that could lead to incorrect domain classification and consequently affect the digit classification. In this section, you are to investigate whether using data from other domains in the training process of the digit classifier results in improved accuracy or not. You'll start by plotting the confusion matrix for the domain classifier and, based on the classifier's predictions, determine which digit classifier should be used for classification. Experiment with different coefficients (how much data from each domain should be used) to train and test your model. Report the results, improvements or lack thereof, and your reasoning for the findings.

To conclude, compare the results of all three phases using an appropriate metric (weighted sum or other) and discuss the accuracy achieved in each part.

This project presents a significant challenge in multi-domain classification and will help you gain insights into handling complex, real-world data scenarios.
