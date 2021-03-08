# Performance assessment with resampling
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-03-08

rm(list = ls())

################################################################################
library(tidymodels)
library(doParallel) # Going to run some model fitting in parallel

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

# For resampling, need to create data folds V = 10
set.seed(55)
ames_folds <- 
  ames_train %>%
  vfold_cv(v = 10)

# We will do resampling, but need an object to make sure the predictions for 
# each resample are kept
keep_pred <- control_resamples(save_pred = TRUE)

# Now apply the workflow to the folds of data
set.seed(130)
rf_results <- 
  rf_workflow %>%
  fit_resamples(resamples = ames_folds, control = keep_pred)

# Show metrics of performance
collect_metrics(rf_results)
# A tibble: 2 x 6
#     .metric .estimator   mean     n std_err .config             
#     <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   0.0691    10 0.00184 Preprocessor1_Model1
#   2 rsq     standard   0.846     10 0.00883 Preprocessor1_Model1

# We want to plot predictions vs. observed, so extract predictions
assess_result <- collect_predictions(rf_results)
assess_result %>%
  ggplot(mapping = aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.15) +
  geom_abline(color = "red") +
  coord_obs_pred() +
  xlab(label = "Observed") +
  ylab(label = "Predicted") +
  ggtitle("Random Forest model")

# Now try resampling approach with a linear regression model
lm_model <- 
  parsnip::linear_reg() %>%
  set_engine("lm")

lm_recipe <- 
  ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

lm_workflow <- 
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(lm_recipe)

# Ready to run the model, but let's try parallelizing it
cl <- parallel::makePSOCKcluster(2) # Two cores
registerDoParallel(cl)

# Now run the model on the V-fold data, 
lm_results <- 
  lm_workflow %>%
  fit_resamples(ames_folds, control = keep_pred)

stopCluster(cl)

# Quick glance at results, shows slightly poorer performance vs. random forest
collect_metrics(lm_results)
# A tibble: 2 x 6
#     .metric .estimator   mean     n std_err .config             
#     <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   0.0750    10 0.00152 Preprocessor1_Model1
#   2 rsq     standard   0.818     10 0.00867 Preprocessor1_Model1

lm_predictions <- collect_predictions(lm_results)

lm_predictions %>%
  ggplot(mapping = aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.15) +
  geom_abline(color = "red") +
  coord_obs_pred() +
  xlab(label = "Observed") +
  ylab(label = "Predicted") +
  ggtitle(label = "Linear regression") +
  theme_bw()
