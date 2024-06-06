# How to Run



## MLFlow

I have created MLFlow to experiment changing the model to track how it's changed

This will allow me to promote the best model to production

### Original Trial

The accuracy on train dataset is:  0.9815
The accuracy on test dataset is:  0.959

This is overfitting

## With Stratify 

The accuracy on train dataset is:  1.0
The accuracy on test dataset is:  0.9635

This is overfitting - still slightly better though 

## Reducing Max Depth

The accuracy on train dataset is:  0.960625
The accuracy on test dataset is:  0.957

F1 score is:  0.8576158940397351
Precision is:  0.9151943462897526
Recall is:  0.8068535825545171