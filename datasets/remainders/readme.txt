These files are not used in either train, test, or validation. 
These were ommitted to prevent data leakage.
E.g. if household number 10 is in the training data of 2018, 
and we evaluate the performance of the model on that household
We cannot have the 2019 data of that same household in the test/validation data
Since that might cause data leakage between train and test/validation Since
the same household is used throughout.