# Random forest model
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-03-05

rm(list = ls())

################################################################################
library(tidymodels)

load(file = "ames.RData")

# Create the random forest model
rf_model <- 
  parsnip::rand_forest(trees = 1000) %>%
  set_engine("ranger") %>% # the ranger package does random forest
  set_mode("regression")   # regression, as opposed to classification

# Make a workflow object
rf_workflow <- 
  workflows::workflow() %>%
  # add_formula basically divides our stuff into response(s) and predictor(s)
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
                Latitude + Longitude) %>%
  add_model(rf_model) # add on that random forest model

# Fit the model on the training data
rf_fit <- 
  fit(rf_workflow, data = ames_train)
# Or, using piping
rf_fit <-
  rf_workflow %>%
  fit(data = ames_train)

# We want to compare the random forest model to the linear regression model, so 
# create the linear regression workflow and fit it on the training data
# Start with model
lm_model <- 
  parsnip::linear_reg() %>%
  set_engine("lm")

# Now build that big, ol recipe. Note, the random forest approach didn't need 
# all this pre-processing. Also note that we are required to pass data when 
# creating a recipe
ames_recipe <- ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

# Create workflow with that model and recipe
lm_workflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(ames_recipe)

# Finally, fit the linear regression model
lm_fit <- 
  lm_workflow %>%
  fit(data = ames_train)

# Create a function that will do the calculations for model performance. In this
# case, use apparent error rate aka resubstitution error rate
estimate_performance <- function(model, dat) {
  # Get the names of objects passed to this function
  cl <- match.call()
  obj_name <- as.character(cl$model)
  data_name <- as.character(cl$dat)
  data_name <- gsub("ames_", "", data_name) # remove "ames_" prefix from data
  
  # Estimate RMSE and R^2
  regression_metrics <- metric_set(rmse, rsq)
  
  # Run the model 
  model %>%
    # Use model to predict values on data (either training or testing data)
    predict(dat) %>%
    # Take those predictions and combine them with observed values
    bind_cols(dat %>% select(Sale_Price)) %>%
    # Calculate performance metrics
    regression_metrics(truth = Sale_Price, estimate = .pred) %>%
    # At this point, we don't need to know what estimator was used
    select(-.estimator) %>%
    mutate(object = obj_name,
           data = data_name)
}

# Now use that function on each of the model fit objects
# Random forest
estimate_performance(model = rf_fit, dat = ames_train)
# # A tibble: 2 x 4
#     .metric .estimate object data 
#     <chr>       <dbl> <chr>  <chr>
#   1 rmse       0.0344 rf_fit train
#   2 rsq        0.965  rf_fit train

# Linear regression
estimate_performance(model = lm_fit, dat = ames_train)
# # A tibble: 2 x 4
#     .metric .estimate object data 
#     <chr>       <dbl> <chr>  <chr>
#   1 rmse       0.0726 lm_fit train
#   2 rsq        0.830  lm_fit train

# Random forest has better rmse

# We can also run this function evaluating performance on the testing data
# Random forest
estimate_performance(model = rf_fit, dat = ames_test)
# # A tibble: 2 x 4
#    .metric .estimate object data 
#     <chr>       <dbl> <chr>  <chr>
#   1 rmse       0.0808 rf_fit test 
#   2 rsq        0.800  rf_fit test 

# Linear regression
estimate_performance(model = lm_fit, dat = ames_test)
# # A tibble: 2 x 4
#     .metric .estimate object data 
#     <chr>       <dbl> <chr>  <chr>
#   1 rmse       0.0882 lm_fit test 
#   2 rsq        0.760  lm_fit test 

# RMSE and rsq pretty similar between the two models when run on the test set