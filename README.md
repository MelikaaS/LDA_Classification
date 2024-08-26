# LDA_Classification
This repository focuses on implementing Linear Discriminant Analysis (LDA) as a classifier on embedded scikit-learn datasets. The goal is to improve the model's performance on these datasets.

## Dataset Overview
This repository utilizes two small embedded scikit-learn datasets: **`load_wine`** and **`load_digits`**.
- **`load_wine`**: A small dataset containing 178 data points with 13 features serving as predictors and three target classes.
- **`load_digits`**: A larger dataset with 1,797 data points, each having 64 features and 10 target classes.

## Operations
### On the `load_wine` Dataset:
1. Fitted the LDA model on the `load_wine` dataset.
2. Extracted the `explained_variance_ratio_` to analyze the contribution of each LDA component.
3. Evaluated the model by calculating the accuracy using **`accuracy_score`**.
4. Conducted a visual analysis using a scatter plot of the LDA components.

### On the `load_digits` Dataset:
1. Fitted the LDA model on the `load_digits` dataset.
2. Extracted the `explained_variance_ratio_` to understand the significance of each LDA component.
3. Evaluated the model's accuracy using **`accuracy_score`**.
4. Conducted a visual analysis using scatter plots of the LDA components.
5. Standardized the data using **`StandardScaler`**.
6. Re-fitted the LDA model on the standardized data to assess the impact of standardization.
7. Recalculated the accuracy using **`accuracy_score`**.
8. Made additional visual observations based on the standardized data.
9. Implemented a pipeline to streamline data standardization and feature selection before fitting the LDA model.
10. Used cross-validation to evaluate the models' performance.
11. Calculated accuracy scores for the pipelined data.
12. Made further visual observations.

## Results
The `load_wine` dataset is relatively small, allowing the LDA model to achieve perfect classification, with an accuracy score of **1.0**. This indicates that the data was perfectly separated into three groups, as demonstrated in the scatter plot of LDA Component 0 against LDA Component 1.

![LDA classification on **`load_wine`** dataset](https://github.com/MelikaaS/LDA_Classification/blob/main/Screenshot%20from%202024-08-26%2011-58-36.jpg)



