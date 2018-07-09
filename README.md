# Multilayer-Perceptron-Training

Trained a Multi-layer Perceptron with one hidden layer having n number of neurons. 'n' can be varied from 5 to 8 (or more). Sigmoid function (activation function) has been used and the learning rate was assumed to be 0.001.

The results (classification accuracy) on the test set for various n values has been printed on terminal and can also be plotted using gnuplot: Plot your results: X-axis = n, Y-axis = classification accuracy. 

Two loss functions, one for each experiment have been implemented. The loss functions are Cross-entropy loss and Sum of Squared Deviation loss.

Similarly, two stopping criterions are to be used, they are:

Stopping Criterion: 
1. You shouldn't fix the number of epochs manually.  Normally, we will use the following formula to update the weights (parameters).
                                                   Wnew  = Wold  + ΔW
The first stopping criteria is ||ΔW|| < ε. where ||ΔW|| is the Euclidian norm of vector ΔW and you may consider ε  as 0.01.
2.   In the second case, you can fix the number of epochs as 100. 
