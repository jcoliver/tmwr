# Ames housing data
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-02-19

rm(list = ls())

# Goal is to predict housing prices
################################################################################
library(tidymodels)

data(ames)

# Sale price (response) is skewed, so transform it to log scale
ames <- ames %>%
  mutate(Sale_Price = log10(Sale_Price))

# Creating testing and training data sets
set.seed(123) # to replicate results in the book
# The testing/training split, stratified on Sale Price for reasonable sampling
ames_split <- initial_split(ames, prop = 0.8, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

save(ames, ames_split, ames_train, ames_test, file = "ames.RData")

?save

# Make the receipe for our model, including predictor transformations and the
# shape of the model
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

# Prep & bake
ames_prepped <- prep(ames_recipe)
ames_train_baked <- bake(ames_prepped, new_data = NULL)
ames_test_baked <- bake(ames_prepped, new_data = ames_test)

# Do simple linear model
lm_fit <- lm(Sale_Price ~ ., data = ames_train_baked)

# Quick glance at some of the results
broom::glance(lm_fit)
broom::tidy(lm_fit)

# And predict values for the first six rows of the test set
predict(lm_fit, newdata = ames_test_baked %>% head())

