README.txt

Julia Hogan
Anthony Todesco
Adam Gordon

May 17, 2014

Bowdoin College
CS3425 Optimization and Uncertainty
Professor Stephen Majercik



This program is Feed Forward Neural Network designed to recognize handwritten digits from the MNIST database. This program is written in Java.

There are two classes than can be used to run this program. The NNTrainer class will train a neural network with some proportion (or all) images from the MNIST database. The NNTester class will test other images from the MNIST database, based on weights from previous executions of NNTrainer.




How to run NNTrainer

NNTrainer takes arguments: learning rate, number of iterations, number of epochs

     Ex: java NNTrainer 0.03 10000 10

To use a random learning rate or a dynamic learning rate, uncomment this.net.randLearningRate() or this.net.dynamicLearningRate(NUM_EPOCHS*ITERATIONS) at lines 102 or 103.



How to run NNTester

NNTester takes arguments: file name (weights), number of hidden nodes, number of output nodes (10 or 4), binary test type (from 0-5)(only matters when number of output nodes is 4).

     Ex: java NNTester WeightFiles/weightsLR03.txt 50 10 0    // decimal test
     Ex: java NNTester WeightFiles/weightsON4.txt 50 4 5      // binary test

If you are using binary test type, you may use 
T.getThreshold();
T.getOppositeThreshold(); 
T.getCompositeThreshold();
in lines 80-82 to retrieve various threshold values. The values will be output into a new file (or overwritten if it already exists) called either threshold.txt, oppositeThreshold.txt, or compositeThreshold.txt.


Copyright Bowdoin College 2014
