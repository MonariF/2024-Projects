breast.cancer <- read.csv("C:/Users/USER/Downloads/breast-cancer.csv")
head(breast.cancer)
str(breast.cancer)

# Load necessary libraries
library(tidyverse)       # For data manipulation and visualization
library(caret)           # For data partitioning and machine learning models
library(e1071)           # For SVM and other ML models
library(corrplot)        # For correlation matrix plot
library(randomForest)    # For random forest model

# Load data (assuming 'breast.cancer' is already loaded in your workspace)

# Objective 1: Exploratory Analysis
# ------------------------------

# 1. Checking structure and summary of the data
str(breast.cancer)
summary(breast.cancer)

# 2. Convert 'diagnosis' to a factor variable
breast.cancer$diagnosis <- factor(breast.cancer$diagnosis, levels = c("B", "M"))

# 3. Plotting correlations between features
# Subset only the numeric columns for correlation analysis
numeric_data <- breast.cancer %>% select(where(is.numeric))
corr_matrix <- cor(numeric_data)
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8, diag = FALSE)

# 4. Boxplots for key features by diagnosis (e.g., radius_mean, texture_mean)
ggplot(breast.cancer, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) +
  geom_boxplot() +
  labs(title = "Distribution of Radius Mean by Diagnosis")

ggplot(breast.cancer, aes(x = diagnosis, y = perimeter_mean, fill = diagnosis)) +
  geom_boxplot() +
  labs(title = "Distribution of Perimeter Mean by Diagnosis")

# Objective 2: Inferential Analysis - Predictive Modeling
# -------------------------------------------------------

# 1. Data Partitioning (80% Train, 20% Test)
set.seed(123)
trainIndex <- createDataPartition(breast.cancer$diagnosis, p = 0.8, list = FALSE)
train_data <- breast.cancer[trainIndex,]
test_data <- breast.cancer[-trainIndex,]

# 2. Logistic Regression Model
logistic_model <- glm(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean +
                        smoothness_mean + compactness_mean + concavity_mean + symmetry_mean, 
                      data = train_data, family = binomial)
summary(logistic_model)

# Predict on test set and evaluate
predictions_logit <- predict(logistic_model, newdata = test_data, type = "response")
predicted_classes_logit <- ifelse(predictions_logit > 0.5, "M", "B")
confusionMatrix(factor(predicted_classes_logit, levels = c("B", "M")), test_data$diagnosis)

# 3. Random Forest Model
rf_model <- randomForest(diagnosis ~ ., data = train_data, importance = TRUE)
print(rf_model)
varImpPlot(rf_model)  # Plot variable importance

# Predict on test set and evaluate
predictions_rf <- predict(rf_model, newdata = test_data)
confusionMatrix(predictions_rf, test_data$diagnosis)

# Objective 3: Identify Significant Predictors
# --------------------------------------------

# Variable importance from Random Forest
importance(rf_model)
varImpPlot(rf_model, main = "Variable Importance")

# Feature selection with Logistic Regression (p-value significance)
summary(logistic_model)

# Using caret's Recursive Feature Elimination (RFE) for feature selection
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
features_rfe <- rfe(train_data[, -c(1, 2)], train_data$diagnosis, sizes = c(1:10), rfeControl = control)
print(features_rfe)

# Conclusion
# Print summary and interpretation of key findings
cat("Exploratory and inferential analysis completed. Significant predictors identified in Random Forest and Logistic Regression.")
