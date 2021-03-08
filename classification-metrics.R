# Binary classification model performance metrics
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-03-03

rm(list = ls())

################################################################################
# 1. Binary classification performance metrics
library(tidymodels)

data(two_class_example)
str(two_class_example)

# 'data.frame':	500 obs. of  4 variables:
#   $ truth    : Factor w/ 2 levels "Class1","Class2": 2 1 2 1 2 1 1 1 2 2 ...
#   $ Class1   : num  0.00359 0.67862 0.11089 0.73516 0.01624 ...
#   $ Class2   : num  0.996 0.321 0.889 0.265 0.984 ...
#   $ predicted: Factor w/ 2 levels "Class1","Class2": 2 1 2 1 2 1 1 1 2 2 ...

# Create confusion matrix
yardstick::conf_mat(two_class_example, truth = truth, estimate = predicted)

# Accuracy
yardstick::accuracy(two_class_example, truth = truth, estimate = predicted)

# Create ROC curve and calculate AUC
# Start with curve; note this does not use the predicted class (a binary 
# outcome), but rather the probability of of being in a particular class, in 
# this example, Class1
two_class_curve <- yardstick::roc_curve(two_class_example, truth, Class1)
# Calculating the area under the curve is similar
yardstick::roc_auc(two_class_example, truth, Class1)
# Plotting the curve uses ggplot, but we only need to call autoplot
autoplot(two_class_curve)

################################################################################
# 2. Multi-class classification performance metrics

# Load a four-class example
data("hpc_cv")
str(hpc_cv)
# 'data.frame':	3467 obs. of  7 variables:
#   $ obs     : Factor w/ 4 levels "VF","F","M","L": 1 1 1 1 1 1 1 1 1 1 ...
#   $ pred    : Factor w/ 4 levels "VF","F","M","L": 1 1 1 1 1 1 1 1 1 1 ...
#   $ VF      : num  0.914 0.938 0.947 0.929 0.942 ...
#   $ F       : num  0.0779 0.0571 0.0495 0.0653 0.0543 ...
#   $ M       : num  0.00848 0.00482 0.00316 0.00579 0.00381 ...
#   $ L       : num  1.99e-05 1.01e-05 5.00e-06 1.56e-05 7.29e-06 ...
#   $ Resample: chr  "Fold01" "Fold01" "Fold01" "Fold01" ...

# Accuracy
yardstick::accuracy(hpc_cv, truth = obs, estimate = pred)
