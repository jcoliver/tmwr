# Workflows examples
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-03-01

rm(list = ls())

################################################################################
# Workflows: preprocessing + modeling

library(tidymodels)

# Loads in data with log-transformed Sale_Price and testing/training split
load(file = "ames.RData")

# Create just the model
lm_model <- linear_reg() %>%  # linear_reg() is a parsnip object
  set_engine("lm")

# Create a workflow object, adding the model
lm_workflow <- workflow() %>%  # from the workflows package
  add_model(lm_model)

# Add a pre-processing step
lm_workflow <- lm_workflow %>%
  add_formula(Sale_Price ~ Longitude + Latitude)

# Workflow objects can use fit() with data
lm_fit <- fit(lm_workflow, ames_train)

# Predict the first five values on the testing data
predict(lm_fit, ames_test %>% slice(1:5))

# Models and preprocessors can updated with update_* functions:
lm_fit_long <- lm_fit %>%
  update_formula(Sale_Price ~ Longitude)

# Instead of formula, can use parsnip recipe (but for this example, need to 
# remove the formula first)

# Start by re-creating the recipe from ames-predictions.R:

# includes predictor transformations and the shape of the model
ames_recipe <- ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  # If there are Neighborhoods only representated by a few rows, collapse them
  # all into an "Other" category
  step_other(Neighborhood, threshold = 0.01) %>%
  # Set up dummy variables for any categorical predictors; in base R, this 
  # would happen in the background, but here we do it explicitly
  step_dummy(all_nominal()) %>%
  # Add interaction term to the receipe; since the categorical Bldg_Type has 
  # been transformed to a series of dummy variables, we use the selector 
  # function starts_with to help out
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  # Predictors may not have linear relationship with response. For example, 
  # latitude is better fit by a spline (a "natural" spline, hence ns), so we 
  # can add that relationship. We'll use a 20-step spline
  step_ns(Latitude, Longitude, deg_free = 20)

# Now use this recipe in the workflow (removing old model first)

lm_workflow <- lm_workflow %>%
  remove_formula() %>%
  add_recipe(ames_recipe)

# Previously, (ames-predictions.R), we used prep, bake, fit/predict, but when 
# using a workflow, we can run fit() (actually, this will be fit-workflow()), 
# which does prep + bake + fit
lm_fit <- fit(lm_workflow, data = ames_train)

# For predicting, we also do not need to run bake & predict separately, the 
# bake is implied when running predict() on a workflow (actually ends up running
# predict-workflow())
predict(lm_fit, new_data = ames_test %>% slice(1:5))

# pull_* functions can extract particular elements from the fit object:
lm_fit %>%
  pull_workflow_fit() %>%
  broom::tidy() %>% # Clean up model fit info
  slice(1:10)

# Assess model performance (of the large model, not the one based solely on 
# Longitude & Latitude)
# Note: not clear why we have to drop the Sale_Price in tibble passed to 
# new_data (excluding the select step does not appear to influence results)
ames_test_predict <- predict(lm_fit,
                             new_data = ames_test %>% select(-Sale_Price))
# Combine these predictions with observed values
ames_test_predict <- bind_cols(ames_test_predict, 
                               ames_test %>% select(Sale_Price))

# Plot predicted vs. observed
ggplot(data = ames_test_predict, mapping = aes(x = Sale_Price, y = .pred)) +
  geom_abline(lty = 2) +
  geom_point(alpha = 0.5) +
  labs(x = "Observed Sale Price (log10)", y = "Predicted Sale Price (log10)") +
  tune::coord_obs_pred() # Sets x & y limits to be the same

# Calculate root mean squared error (RMSE)
rmse(ames_test_predict, truth = Sale_Price, estimate = .pred)

# To calculate multiple metrics, create a metric set via yardstick::metric_set
# Add RMSE, R^2, and mean absolute error
ames_metrics <- metric_set(rmse, rsq, mae)
ames_metrics(ames_test_predict, truth = Sale_Price, estimate = .pred)
