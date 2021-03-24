# Model performance comparison with resampling
# Jeff Oliver
# jcoliver@arizona.edu
# 2021-03-24

################################################################################
library(tidymodels)

load(file = "ames.RData")

# Going to fit and compare four models with resampling:
# 1. Original linear regression model (with splines)
# 2. Simpler linear regression model (sans splines)
# 3. Random forest model

# So we tell R to save predictions from resampling events
keep_pred <- control_resamples(save_pred = TRUE)

########################################
# 1. Original linear regression model (with splines)
# Create linear regression model
lm_model <- 
  parsnip::linear_reg() %>%
  set_engine("lm")

# Set up recipe, which establishes data pre-processing and the model in terms 
# of which variables are predictors and which is/are response variable(s) as 
# well as any interaction terms and fancy modeling (in this case, a spline for
# longitude and latitude)
lm_recipe <- 
  ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

# Set up workflow, which is basically model + recipe
lm_workflow <- 
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(lm_recipe)

# Do the resampling with this linear model workflow
lm_splines_fit <- 
  lm_workflow %>%
  fit_resamples(resamples = ames_folds, control = keep_pred)

########################################
# 2. Simpler linear regression model (sans splines)

# Create the recipe, which just leaves off the step_ns piece
lm_simple_recipe <- 
  ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_"))

# Fit the model, creating the workflow (temporarily) in the process; note we can
# use the same model object (lm_model), as this only specifies the engine (lm).
lm_simple_fit <- 
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(lm_simple_recipe) %>%
  fit_resamples(resamples = ames_folds, control = keep_pred)
  
# Comparing the two linear regression models
collect_metrics(lm_splines_fit)
#     .metric .estimator   mean     n std_err .config             
#     <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   0.0752    10 0.00241 Preprocessor1_Model1
#   2 rsq     standard   0.817     10 0.00897 Preprocessor1_Model1
collect_metrics(lm_simple_fit)
#     .metric .estimator   mean     n std_err .config             
#     <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   0.0769    10 0.00229 Preprocessor1_Model1
#   2 rsq     standard   0.809     10 0.00845 Preprocessor1_Model1

########################################
# 3. Random forest model
# Create the random forest model
rf_model <- 
  parsnip::rand_forest(trees = 1000) %>%
  set_engine("ranger") %>% # the ranger package does random forest
  set_mode("regression")   # regression, as opposed to classification

# The random forest workflow doesn't require a bunch of pre-processing, so we 
# skip the receipe creation step
rf_workflow <- 
  workflows::workflow() %>%
  # add_formula basically divides our stuff into response(s) and predictor(s)
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
                Latitude + Longitude) %>%
  add_model(rf_model) # add on that random forest model

# Fit the random forest model with the 10 resamples
rf_fit <- 
  rf_workflow %>%
  fit_resamples(resamples = ames_folds, control = keep_pred)

collect_metrics(rf_fit)
#     .metric .estimator   mean     n std_err .config             
#     <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
#   1 rmse    standard   0.0687    10 0.00249 Preprocessor1_Model1
#   2 rsq     standard   0.848     10 0.00969 Preprocessor1_Model1

# In each model fit, they all used the same resampling approach (a good thing), 
# taking the same subset of the training data to fit the model and evaluating 
# performance (RMSE and Rsq) on the remaining data (validation subset). Because 
# of this the performance metrics across models should account for the non-
# independence between the three different model fits. i.e. results from the 
# first resample spline model are not independent of the model fits from the 
# first resample of the random forest model.

# Going to compare the models, using Rsq as measure of performance. 
# Extract Rsq from each
lm_simple_rsq <- 
  collect_metrics(lm_simple_fit, summarize = FALSE) %>%
  filter(.metric == "rsq") %>%
  select(id, `no splines` = .estimate)

lm_splines_rsq <-
  collect_metrics(lm_splines_fit, summarize = FALSE) %>%
  filter(.metric == "rsq") %>%
  select(id, `splines` = .estimate)

rf_rsq <-
  collect_metrics(rf_fit, summarize = FALSE) %>%
  filter(.metric == "rsq") %>%
  select(id, `random forest` = .estimate)

# Join all these tibbles together
rsq_estimates <-
  lm_simple_rsq %>%
  inner_join(lm_splines_rsq, by = "id") %>%
  inner_join(rf_rsq, by = "id")
#    id     `no splines` splines `random forest`
#    <chr>         <dbl>   <dbl>           <dbl>
# 1  Fold01        0.747   0.751           0.778
# 2  Fold02        0.839   0.853           0.887
# 3  Fold03        0.794   0.801           0.827
# 4  Fold04        0.793   0.802           0.833
# 5  Fold05        0.812   0.819           0.864
# 6  Fold06        0.821   0.822           0.864
# 7  Fold07        0.820   0.835           0.845
# 8  Fold08        0.805   0.816           0.852
# 9  Fold09        0.827   0.836           0.855
# 10 Fold10        0.834   0.838           0.876

# See if Rsq are correlated
corrr::correlate(rsq_estimates %>% select(-id))
#     term          `no splines` splines `random forest`
#     <chr>                <dbl>   <dbl>           <dbl>
#   1 no splines          NA       0.989           0.960
#   2 splines              0.989  NA               0.938
#   3 random forest        0.960   0.938          NA 

# To account for this non-independence, could do a random effects model, where 
# model type is fixed effect, fold is random effect, and Rsq is response.

# Could also do pairwise comparisons, where difference in Rsq between models 
# (within folds) is the response variable (test for significant departure from 
# zero). A pairwise t-test provides the same analysis.

# Alternatively, can use Bayesian approach to generate posterior distributions 
# of Rsq and include fold as a random intercept effect. The model is 
# Rsq ~ B0 + B1 * x1 + B2 * x2 + bi
# Where
#    B0 is intercept
#    x1 is dummy variable for spline model
#    x2 is dummy variable for random forest model
#    bi is random effect of fold
