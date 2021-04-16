# Bayesian comparison of four models
# Jeff Oliver
# jcoliver@arizona.edu
# 2021-04-12

rm(list = ls())

################################################################################
library(tidymodels)
library(tidyposterior)
library(rstanarm)
library(forcats)
library(workflowsets)

# Going to compare four models: three linear regression, plus random forest

load(file = "ames.RData")

# For the linear models, we have
# 1. A simple linear model
# 2. An interaction model, with interaction between living area & building type
# 3. A spline model, which is the interaction model plus a spline for lat & lng

# We can build these different models iteratively (because they are nested)
simple_recipe <- ames_train %>%
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal())

# Model 2 includes an interaction between living area & building type
interact_recipe <- simple_recipe %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_"))

# Model 3 includes spline for lat & lng
spline_recipe <- interact_recipe %>%
  step_ns(Latitude, Longitude, deg_free = 50)

# Need to bundle these together in a list before sending to workflowset
recipes <- list(simple = simple_recipe,
                interact = interact_recipe,
                splines = spline_recipe)

# Have to indicate the engine we will use for the workflow
lm_model <- 
  parsnip::linear_reg() %>%
  set_engine("lm")

lm_models <- workflow_set(preproc = recipes,
                          models = list(lm = lm_model),
                          cross = FALSE)

# Set what gets saved (here we save predictions AND the workflow) in resampling
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

# Resample the models
lm_models <- lm_models %>%
  workflow_map(fn = "fit_resamples",   # function to execute
               seed = 1001, 
               verbose = TRUE,
               resamples = ames_folds, # resampling folds
               control = keep_pred)

# Pull out the RMSE
collect_metrics(lm_models) %>%
  filter(.metric == "rmse")

# We can include the random forest model, but we'll need to make it and be 
# sure to set save_workflow = TRUE in resampling step
# Copied from performance-resampling-comparison.R

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

# Fit the random forest model with the 10 resamples (that are defined by 
# ames_folds). Note this variable is called rf_res on TMWR.org
rf_fit <- 
  rf_workflow %>%
  fit_resamples(resamples = ames_folds, control = keep_pred)

# Now we can add that random forest to the three linear models
four_models <- as_workflow_set(random_forest = rf_fit) %>%
  bind_rows(lm_models)

# Plot R^2 for each model
autoplot(four_models, metric = "rsq")

# Not much improvement in linear models as they increase in complexity
# Also, R^2 is likely correlated within resamples, across models
# Pull out each estimate and see what correlation is like between models for 
# the same resample
rsq_ind_est <- collect_metrics(four_models, summarize = FALSE) %>%
  filter(.metric == "rsq")

rsq_wide <- rsq_ind_est %>%
  select(wflow_id, .estimate, id) %>%
  pivot_wider(id_cols = id,
              names_from = wflow_id,
              values_from = .estimate)
corrr::correlate(rsq_wide %>% select(-id), quiet = TRUE)

# We can also plot each reasample as a line
rsq_ind_est %>%
  mutate(wflow_id = reorder(wflow_id, .estimate)) %>%
  ggplot(mapping = aes(x = wflow_id, y = .estimate, group = id, col = id)) +
  geom_line() +
  theme(legend.position = "none")

# Now we can finally to Bayesian analysis, comparing the four models
# Had to update to tidyposterior 0.1.0 (from 0.0.3) to get this to work
rsq_anova <- perf_mod(four_models,
                      metric = "rsq",
                      prior_intercept = rstanarm::student_t(df = 1),
                      chains = 4,
                      iter = 5000,
                      seed = 1102)

# Extract posteriors
model_posterior <- rsq_anova %>%
  tidy(see = 1003)

# Plot all the posteriors
model_posterior %>%
  mutate(model = forcats::fct_inorder(model)) %>%
  ggplot(mapping = aes(x = posterior)) +
  geom_histogram(bins = 50) +
  facet_wrap(~ model, ncol = 1) +
  xlab(label = "Posterior rsq") +
  theme_bw()

# Use autoplot again to see the metric (R^2)
autoplot(rsq_anova)

# We can contrast the differences in means from the posterior distribution
# Just looking at the simple model vs. the splines + interaction model
rsq_diff <- contrast_models(x = rsq_anova,
                            list_1 = "splines_lm",
                            list_2 = "simple_lm",
                            seed = 1003)
# and plot
rsq_diff %>%
  as_tibble() %>%
  ggplot(mapping = aes(x = difference)) +
  geom_histogram(bins = 100) +
  geom_vline(xintercept = 0, lty = 2) +
  theme_bw()
  
# Compare means between simple vs. splines + interaction models
summary(rsq_diff) %>%
  select(-starts_with("pract"))

# We can compare all four models for practical significance (i.e. must be 
# at least 2% better to be meaningfully different)
autoplot(rsq_anova, type = "ROPE", size = 0.02)
