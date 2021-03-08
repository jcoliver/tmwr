# Modeling with parsinp
# Jeffrey C. Oliver
# jcoliver@email.arizona.edu
# 2021-02-22

rm(list = ls())

################################################################################
library(tidymodels)

load(file = "ames.RData")

lm_model <- linear_reg() %>%
  set_engine("lm")

# Two different ways of specifying a model with parsnip.
# The first (fit), uses formula specification, i.e. y ~ x; if there are nominal
# (categorical) predictors in x, dummy variables will be automatically created
lm_fit <- lm_model %>%
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

# Second approach (fit_xy) uses x-y specification, the former being a matrix, 
# the latter being a vector; will *not* create dummy variables for x (although
# it seems to handle categorical predictors just fine...)
lm_fit_xy <- lm_model %>%
  fit_xy(x = ames_train %>% select(Longitude, Latitude),
         y = ames_train %>% pull(Sale_Price))

# Extracting results to use with summary
lm_fit %>%
  purrr::pluck("fit") %>%
  summary()

# Extract results in a form we can use
broom::tidy(lm_fit)

