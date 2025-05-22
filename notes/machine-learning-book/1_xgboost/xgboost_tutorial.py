# %% [raw]
# ---
# title: "EXERCISE: XGBoost"
# categories: Machine Learning
# date: 05-22-2025
# ---

# %% [markdown]
# # XGBoost: A Complete Learning Journey
# 
# Welcome to this comprehensive tutorial on XGBoost! This notebook is designed to take you from zero knowledge to building real-world models.
# 
# ## 📚 Learning Path:
# 
# **Part 1: Foundations (Theory & Practice)**
# - Lesson 1: What is Machine Learning Boosting?
# - Lesson 2: Understanding Decision Trees
# - Lesson 3: From Trees to Ensemble Methods
# - Lesson 4: Gradient Boosting Explained
# - Lesson 5: Enter XGBoost
# - Lesson 6: XGBoost Parameters Deep Dive
# - Lesson 7: Preventing Overfitting
# 
# **Part 2: Hands-On Project**
# - Building a Complete Credit Risk Model

# %% [markdown]
# ## 🚀 Let's Start: Setting Up Our Environment

# %%
# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

print("✅ Environment ready! Let's learn XGBoost!")

# %% [markdown]
# ---
# # Part 1: Foundations
# 
# ## Lesson 1: What is Machine Learning Boosting? 🤔
# 
# Imagine you're trying to become an expert at predicting weather. Instead of relying on one meteorologist, wouldn't it be better to:
# 1. Ask multiple meteorologists
# 2. Have each one learn from the mistakes of the previous ones
# 3. Combine all their predictions
# 
# That's **boosting** in a nutshell!
# 
# ### Key Concept:
# **Boosting** = Combining many "weak" learners to create one "strong" learner

# %%
# Let's visualize this concept with a simple example
# Create a non-linear dataset that's hard for a single model

# Generate data
np.random.seed(42)
X_demo = np.linspace(-3, 3, 300).reshape(-1, 1)
y_demo = np.sin(2 * X_demo).ravel() + np.sin(4 * X_demo).ravel() + np.random.normal(0, 0.3, X_demo.shape[0])

# Visualize the challenge
plt.figure(figsize=(10, 6))
plt.scatter(X_demo, y_demo, alpha=0.5, s=20)
plt.title("Our Challenge: Predict this Complex Pattern", fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### 🧠 Think About It:
# Could a single straight line predict this pattern well? How about a simple curve? This is why we need boosting!

# %% [markdown]
# ## Lesson 2: Understanding Decision Trees 🌳
# 
# Before we dive into boosting, let's understand the building block: **Decision Trees**.
# 
# A decision tree makes predictions by asking a series of yes/no questions:
# - Is X > 5? → Yes → Is Y < 3? → Yes → Predict Class A
# 
# Let's see this in action:

# %%
# Create a simple classification dataset
X_tree, y_tree = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, 
                                     random_state=42, flip_y=0.1)

# Visualize the data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_tree[:, 0], X_tree[:, 1], c=y_tree, cmap='viridis', s=50, edgecolor='k')
plt.title("Our Classification Problem")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(scatter, label='Class')

# Train a simple decision tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_tree, y_tree)

# Visualize decision boundaries
plt.subplot(1, 2, 2)
x_min, x_max = X_tree[:, 0].min() - 1, X_tree[:, 0].max() + 1
y_min, y_max = X_tree[:, 1].min() - 1, X_tree[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_tree[:, 0], X_tree[:, 1], c=y_tree, cmap='viridis', s=50, edgecolor='k')
plt.title("Decision Tree Boundaries")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 Exercise 1: Tree Depth Impact
# 
# **Task**: Modify the tree depth and observe how it affects the decision boundaries. What happens with depth=1 vs depth=10?

# %%
# Exercise 1: Try different tree depths
depths_to_try = [1, 3, 10]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, depth in enumerate(depths_to_try):
    # TODO: Train a tree with the specified depth
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_tree, y_tree)
    
    # Visualization code (uncomment when you've trained the tree)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    axes[idx].scatter(X_tree[:, 0], X_tree[:, 1], c=y_tree, cmap='viridis', s=50, edgecolor='k')
    axes[idx].set_title(f'Depth = {depth}')
    
    # Placeholder
    # axes[idx].text(0.5, 0.5, f'TODO: Depth={depth}', ha='center', va='center', transform=axes[idx].transAxes)
    # axes[idx].set_title(f'Depth = {depth}')

plt.tight_layout()
plt.show()

print("💭 What do you notice? Shallow trees are simple (underfitting), deep trees are complex (overfitting)!")

# %% [markdown]
# ## Lesson 3: From Single Trees to Ensemble Methods 🌲🌲🌲
# 
# A single tree can be:
# - **Too simple** (underfitting): Misses important patterns
# - **Too complex** (overfitting): Memorizes noise
# 
# **Solution**: Use multiple trees and combine their predictions!
# 
# There are two main approaches:
# 1. **Bagging** (Random Forest): Trees learn independently in parallel
# 2. **Boosting** (XGBoost): Trees learn sequentially, each fixing the errors of the previous

# %%
# Let's demonstrate the difference between a single tree and boosting
from sklearn.ensemble import GradientBoostingRegressor

# Use our sine wave data from earlier
X_train, X_test = X_demo[:200], X_demo[200:]
y_train, y_test = y_demo[:200], y_demo[200:]

# Train different models
single_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
single_tree.fit(X_train, y_train)

boosted_trees = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
boosted_trees.fit(X_train, y_train)

# Predictions
pred_single = single_tree.predict(X_test)
pred_boosted = boosted_trees.predict(X_test)

# Visualize
plt.figure(figsize=(15, 5))

# Single tree
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, alpha=0.5, s=20, label='Training data')
plt.plot(X_test, pred_single, 'r-', linewidth=2, label='Single tree prediction')
plt.title('Single Decision Tree')
plt.legend()
plt.grid(True, alpha=0.3)

# Boosted trees
plt.subplot(1, 3, 2)
plt.scatter(X_train, y_train, alpha=0.5, s=20, label='Training data')
plt.plot(X_test, pred_boosted, 'g-', linewidth=2, label='Boosted trees prediction')
plt.title('Gradient Boosting (50 trees)')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparison
plt.subplot(1, 3, 3)
plt.plot(X_test, y_test, 'k-', linewidth=2, label='True function', alpha=0.7)
plt.plot(X_test, pred_single, 'r--', linewidth=2, label='Single tree', alpha=0.7)
plt.plot(X_test, pred_boosted, 'g--', linewidth=2, label='Boosted trees', alpha=0.7)
plt.title('Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate errors
mse_single = mean_squared_error(y_test, pred_single)
mse_boosted = mean_squared_error(y_test, pred_boosted)
print(f"📊 Single Tree MSE: {mse_single:.4f}")
print(f"📊 Boosted Trees MSE: {mse_boosted:.4f}")
print(f"🎯 Improvement: {(1 - mse_boosted/mse_single)*100:.1f}%")

# %% [markdown]
# ## Lesson 4: How Gradient Boosting Works 🔧
# 
# Let's break down the magic of gradient boosting step by step:
# 
# ### The Algorithm:
# 1. **Start** with a simple prediction (like the mean)
# 2. **Calculate** the errors (residuals)
# 3. **Train** a new tree to predict these errors
# 4. **Add** this tree's predictions (scaled down) to improve our model
# 5. **Repeat** steps 2-4
# 
# Think of it like improving your aim in darts:
# - First throw: You miss by a lot
# - Second throw: You adjust based on how much you missed
# - Third throw: You make smaller adjustments
# - Eventually: You hit the bullseye!

# %%
# Let's build gradient boosting from scratch to understand it!
class SimpleGradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
    
    def fit(self, X, y):
        # Step 1: Initial prediction (mean)
        self.init_prediction = np.mean(y)
        predictions = np.full(len(y), self.init_prediction)
        
        # Store predictions at each iteration for visualization
        self.iteration_predictions = [predictions.copy()]
        
        for i in range(self.n_estimators):
            # Step 2: Calculate residuals
            residuals = y - predictions
            
            # Step 3: Train tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42+i)
            tree.fit(X, residuals)
            
            # Step 4: Update predictions
            predictions += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
            self.iteration_predictions.append(predictions.copy())
    
    def predict(self, X):
        predictions = np.full(len(X), self.init_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Train our simple gradient boosting
X_simple = X_demo[:200].reshape(-1, 1)
y_simple = y_demo[:200]

gb_model = SimpleGradientBoosting(n_estimators=5, learning_rate=0.3, max_depth=3)
gb_model.fit(X_simple, y_simple)

# Visualize the boosting process
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(6):
    ax = axes[i]
    ax.scatter(X_simple, y_simple, alpha=0.5, s=20, label='True data')
    
    if i == 0:
        ax.axhline(y=gb_model.init_prediction, color='r', linestyle='--', 
                   label=f'Initial prediction (mean={gb_model.init_prediction:.2f})')
        ax.set_title('Step 0: Initial Prediction')
    else:
        # Sort for smooth line plot
        sort_idx = np.argsort(X_simple.ravel())
        ax.plot(X_simple[sort_idx], gb_model.iteration_predictions[i-1][sort_idx], 
                'r-', linewidth=2, label=f'After {i} trees')
        ax.set_title(f'After {i} Tree(s)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.show()

print("🎯 See how each tree improves the prediction? That's the power of boosting!")

# %% [markdown]
# ### 💡 Exercise 2: Learning Rate Experiment
# 
# The learning rate controls how much we trust each new tree. Let's experiment!

# %%
# Exercise 2: Impact of learning rate
learning_rates = [0.01, 0.1, 0.5, 1.0]

plt.figure(figsize=(15, 10))

for idx, lr in enumerate(learning_rates):
    plt.subplot(2, 2, idx + 1)
    
    # TODO: Train a gradient boosting model with the specified learning rate
    # model = SimpleGradientBoosting(n_estimators=20, learning_rate=lr, max_depth=3)
    # model.fit(X_simple, y_simple)
    # predictions = model.predict(X_simple)
    
    # TODO: Plot the results
    # plt.scatter(X_simple, y_simple, alpha=0.5, s=20)
    # sort_idx = np.argsort(X_simple.ravel())
    # plt.plot(X_simple[sort_idx], predictions[sort_idx], 'r-', linewidth=2)
    # plt.title(f'Learning Rate = {lr}')
    # plt.grid(True, alpha=0.3)
    
    # Placeholder
    plt.text(0.5, 0.5, f'TODO: LR={lr}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(f'Learning Rate = {lr}')

plt.tight_layout()
plt.show()

print("💭 Question: Which learning rate gives the smoothest predictions? Why?")

# %% [markdown]
# ## Lesson 5: Enter XGBoost! 🚀
# 
# XGBoost (eXtreme Gradient Boosting) is gradient boosting on steroids! It adds:
# 
# ### 1. **Speed Improvements**:
# - Parallel processing
# - Cache optimization
# - Efficient data structures
# 
# ### 2. **Accuracy Improvements**:
# - Regularization (L1 & L2)
# - Better handling of missing values
# - More intelligent tree pruning
# 
# ### 3. **Flexibility**:
# - Custom objective functions
# - Built-in cross-validation
# - Early stopping
# 
# Let's see XGBoost in action!

# %%
# Compare regular gradient boosting with XGBoost
from time import time

# Create a larger dataset to see speed differences
X_large, y_large = make_classification(n_samples=5000, n_features=20, 
                                       n_informative=15, random_state=42)
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
    X_large, y_large, test_size=0.3, random_state=42
)

# Regular Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
start_time = time()
gb_sklearn = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_sklearn.fit(X_train_large, y_train_large)
gb_time = time() - start_time
gb_score = gb_sklearn.score(X_test_large, y_test_large)

# XGBoost
start_time = time()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_large, y_train_large)
xgb_time = time() - start_time
xgb_score = xgb_model.score(X_test_large, y_test_large)

# Results
results_df = pd.DataFrame({
    'Model': ['Gradient Boosting', 'XGBoost'],
    'Training Time (s)': [gb_time, xgb_time],
    'Accuracy': [gb_score, xgb_score]
})

# Visualize comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training time
ax1.bar(results_df['Model'], results_df['Training Time (s)'], color=['blue', 'red'])
ax1.set_ylabel('Training Time (seconds)')
ax1.set_title('Training Speed Comparison')

# Accuracy
ax2.bar(results_df['Model'], results_df['Accuracy'], color=['blue', 'red'])
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Performance')
ax2.set_ylim(0.9, 1.0)

plt.tight_layout()
plt.show()

print(results_df)
print(f"\n⚡ XGBoost is {gb_time/xgb_time:.1f}x faster!")

# %% [markdown]
# ## Lesson 6: XGBoost Parameters Deep Dive 🎛️
# 
# XGBoost has many parameters, but let's focus on the most important ones:
# 
# ### 🌳 Tree Parameters:
# - `max_depth`: How deep can trees grow? (default: 6)
# - `min_child_weight`: Minimum data in leaf nodes (default: 1)
# - `gamma`: Minimum loss reduction for split (default: 0)
# 
# ### 📚 Boosting Parameters:
# - `n_estimators`: Number of trees (default: 100)
# - `learning_rate`: Step size (default: 0.3)
# - `subsample`: Fraction of data per tree (default: 1.0)
# 
# ### 🛡️ Regularization Parameters:
# - `reg_alpha`: L1 regularization (default: 0)
# - `reg_lambda`: L2 regularization (default: 1)
# - `colsample_bytree`: Fraction of features per tree (default: 1.0)

# %%
# Interactive parameter exploration
# Let's see how max_depth and n_estimators affect performance

# Create a dataset
X_param, y_param = make_classification(n_samples=1000, n_features=10, 
                                       n_informative=8, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_param, y_param, test_size=0.3, random_state=42
)

# Parameter grid
max_depths = [2, 4, 6, 8, 10]
n_estimators_list = [10, 50, 100, 200]

# Store results
results = np.zeros((len(max_depths), len(n_estimators_list)))

for i, max_depth in enumerate(max_depths):
    for j, n_est in enumerate(n_estimators_list):
        model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_est,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_p, y_train_p, verbose=False)
        results[i, j] = model.score(X_test_p, y_test_p)

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(results, annot=True, fmt='.3f', 
            xticklabels=n_estimators_list,
            yticklabels=max_depths,
            cmap='YlOrRd')
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.title('XGBoost Performance: Impact of Key Parameters')
plt.show()

print("🎯 Notice: More trees and deeper trees generally improve performance, but with diminishing returns!")

# %% [markdown]
# ### 💡 Exercise 3: Regularization Experiment
# 
# Regularization helps prevent overfitting. Let's see it in action!

# %%
# Exercise 3: Effect of regularization
# Create an overfitting scenario with few samples and many features
X_overfit, y_overfit = make_classification(n_samples=200, n_features=50, 
                                           n_informative=10, random_state=42)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_overfit, y_overfit, test_size=0.5, random_state=42
)

# TODO: Train two models - one without regularization, one with
# model_no_reg = xgb.XGBClassifier(
#     n_estimators=100,
#     max_depth=10,
#     reg_alpha=0,
#     reg_lambda=0,
#     random_state=42
# )

# model_with_reg = xgb.XGBClassifier(
#     n_estimators=100,
#     max_depth=10,
#     reg_alpha=1,
#     reg_lambda=1,
#     random_state=42
# )

# TODO: Fit both models and compare train vs test accuracy
# Plot the results showing overfitting vs regularized model

print("TODO: Complete this exercise to see how regularization prevents overfitting!")

# %% [markdown]
# ## Lesson 7: Preventing Overfitting in XGBoost 🛡️
# 
# XGBoost provides several techniques to prevent overfitting:
# 
# 1. **Early Stopping**: Stop training when validation score stops improving
# 2. **Cross-Validation**: Built-in CV for parameter tuning
# 3. **Regularization**: L1/L2 penalties
# 4. **Subsampling**: Use random subsets of data/features

# %%
# Demonstrate early stopping
X_early, y_early = make_classification(n_samples=2000, n_features=20, 
                                       n_informative=15, random_state=42)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_early, y_early, test_size=0.3, random_state=42
)

# Train with early stopping
model_early = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=42)

# Set up evaluation set
eval_set = [(X_train_e, y_train_e), (X_test_e, y_test_e)]

# Fit with early stopping
model_early.fit(
    X_train_e, y_train_e,
    eval_set=eval_set,
    eval_metric='logloss',
    early_stopping_rounds=10,
    verbose=False
)

# Plot training history
results = model_early.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.axvline(x=model_early.best_iteration, color='red', linestyle='--', 
            label=f'Early Stop (iteration {model_early.best_iteration})')
plt.xlabel('Boosting Iterations')
plt.ylabel('Log Loss')
plt.title('Early Stopping in XGBoost')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"🛑 Training stopped at iteration {model_early.best_iteration} (out of 500 planned)")
print(f"📊 Best test score: {model_early.best_score:.4f}")

# %% [markdown]
# ---
# # Part 2: Hands-On Project 🏗️
# 
# ## Credit Risk Assessment Model
# 
# Now let's apply everything we've learned to build a real-world credit risk model!
# 
# **Scenario**: You work for a bank and need to predict whether loan applicants will default.
# 
# **Goal**: Build an XGBoost model that can accurately identify high-risk applicants while minimizing false rejections of good customers.

# %% [markdown]
# ## Step 1: Data Generation and Understanding 📊
# 
# First, let's create a realistic credit dataset with meaningful features.

# %%
# Generate synthetic credit data
np.random.seed(42)
n_customers = 10000

# Create realistic features
credit_data = pd.DataFrame({
    # Demographics
    'age': np.random.normal(40, 12, n_customers).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.6, n_customers),
    'employment_years': np.random.exponential(7, n_customers).clip(0, 40),
    
    # Credit history
    'credit_score': np.random.normal(700, 100, n_customers).clip(300, 850).astype(int),
    'num_credit_cards': np.random.poisson(3, n_customers),
    'num_loans': np.random.poisson(2, n_customers),
    
    # Current loan details
    'loan_amount': np.random.lognormal(10, 0.8, n_customers),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_customers),
    'interest_rate': np.random.normal(10, 3, n_customers).clip(3, 20),
    
    # Financial behavior
    'debt_to_income': np.random.beta(2, 5, n_customers) * 0.8,
    'missed_payments': np.random.poisson(0.5, n_customers),
    'credit_utilization': np.random.beta(2, 5, n_customers),
    
    # Account information
    'checking_balance': np.random.lognormal(8, 1.5, n_customers),
    'savings_balance': np.random.lognormal(9, 1.8, n_customers),
    'months_since_last_delinquent': np.random.exponential(24, n_customers).clip(0, 100)
})

# Create realistic default probability based on features
default_probability = (
    0.05 +  # Base rate
    0.15 * (credit_data['credit_score'] < 600) +
    0.10 * (credit_data['debt_to_income'] > 0.4) +
    0.10 * (credit_data['missed_payments'] > 2) +
    0.08 * (credit_data['credit_utilization'] > 0.8) +
    0.07 * (credit_data['income'] < 30000) +
    0.05 * (credit_data['checking_balance'] < 1000) +
    0.05 * (credit_data['employment_years'] < 1) +
    np.random.normal(0, 0.05, n_customers)
).clip(0, 1)

# Generate default labels
credit_data['default'] = (np.random.random(n_customers) < default_probability).astype(int)

# Display basic statistics
print("📊 Dataset Overview:")
print(f"Total customers: {len(credit_data):,}")
print(f"Default rate: {credit_data['default'].mean():.1%}")
print("\n📈 Feature Statistics:")
print(credit_data.describe())

# %% [markdown]
# ## Step 2: Exploratory Data Analysis (EDA) 🔍
# 
# Let's understand our data better through visualization.

# %%
# EDA visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# 1. Default rate distribution
credit_data['default'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Default Distribution')
axes[0].set_xticklabels(['No Default', 'Default'], rotation=0)
axes[0].set_ylabel('Count')

# 2. Credit score by default status
credit_data.boxplot(column='credit_score', by='default', ax=axes[1])
axes[1].set_title('Credit Score by Default Status')
axes[1].set_xticklabels(['No Default', 'Default'])

# 3. Income distribution by default
credit_data[credit_data['income'] < 200000].boxplot(column='income', by='default', ax=axes[2])
axes[2].set_title('Income by Default Status')
axes[2].set_xticklabels(['No Default', 'Default'])

# 4. Debt-to-income ratio
credit_data.boxplot(column='debt_to_income', by='default', ax=axes[3])
axes[3].set_title('Debt-to-Income Ratio')
axes[3].set_xticklabels(['No Default', 'Default'])

# 5. Missed payments impact
missed_payments_default = credit_data.groupby('missed_payments')['default'].mean()
missed_payments_default.plot(kind='bar', ax=axes[4], color='orange')
axes[4].set_title('Default Rate by Missed Payments')
axes[4].set_xlabel('Number of Missed Payments')
axes[4].set_ylabel('Default Rate')

# 6. Feature correlation heatmap
numeric_cols = ['credit_score', 'income', 'debt_to_income', 'missed_payments', 
                'credit_utilization', 'default']
correlation_matrix = credit_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[5], 
            fmt='.2f', square=True)
axes[5].set_title('Feature Correlations')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 Key Insights:
# - Credit score is strongly correlated with default (negative correlation)
# - Missed payments are a strong indicator of default
# - Debt-to-income ratio affects default probability
# - Income alone is not a strong predictor

# %% [markdown]
# ## Step 3: Feature Engineering 🔧
# 
# Let's create some powerful features that might help our model.

# %%
# Feature engineering
credit_data_eng = credit_data.copy()

# Create new features
credit_data_eng['loan_to_income_ratio'] = credit_data_eng['loan_amount'] / credit_data_eng['income']
credit_data_eng['total_debt'] = credit_data_eng['debt_to_income'] * credit_data_eng['income']
credit_data_eng['monthly_payment'] = (credit_data_eng['loan_amount'] * 
                                      (credit_data_eng['interest_rate'] / 100 / 12)) / \
                                     (1 - (1 + credit_data_eng['interest_rate'] / 100 / 12) ** 
                                      (-credit_data_eng['loan_term_months']))
credit_data_eng['payment_to_income'] = credit_data_eng['monthly_payment'] / (credit_data_eng['income'] / 12)
credit_data_eng['financial_health_score'] = (
    credit_data_eng['credit_score'] / 850 * 0.3 +
    (1 - credit_data_eng['debt_to_income']) * 0.3 +
    (1 - credit_data_eng['credit_utilization']) * 0.2 +
    np.log1p(credit_data_eng['savings_balance']) / 15 * 0.2
)
credit_data_eng['risk_category'] = pd.cut(
    credit_data_eng['credit_score'], 
    bins=[0, 580, 670, 740, 850],
    labels=['Poor', 'Fair', 'Good', 'Excellent']
)

# Show new features
print("🔧 New engineered features:")
new_features = ['loan_to_income_ratio', 'total_debt', 'monthly_payment', 
                'payment_to_income', 'financial_health_score']
print(credit_data_eng[new_features].describe())

# %% [markdown]
# ## Step 4: Data Preprocessing 🧹
# 
# Prepare the data for XGBoost modeling.

# %%
# Preprocessing
# Handle categorical variables
categorical_cols = ['risk_category']
credit_data_encoded = pd.get_dummies(credit_data_eng, columns=categorical_cols, prefix='risk')

# Separate features and target
feature_cols = [col for col in credit_data_encoded.columns if col not in ['default']]
X = credit_data_encoded[feature_cols]
y = credit_data_encoded['default']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("🔄 Data split:")
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"\nDefault rates:")
print(f"Train: {y_train.mean():.1%}")
print(f"Validation: {y_val.mean():.1%}")
print(f"Test: {y_test.mean():.1%}")

# %% [markdown]
# ## Step 5: Baseline Model 📊
# 
# Let's start with a simple XGBoost model to establish a baseline.

# %%
# Baseline model
baseline_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_estimators=100
)

# Train
baseline_model.fit(X_train, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Evaluate
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

baseline_auc = roc_auc_score(y_test, y_pred_proba_baseline)
baseline_ap = average_precision_score(y_test, y_pred_proba_baseline)

print("📊 Baseline Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"ROC AUC: {baseline_auc:.3f}")
print(f"Average Precision: {baseline_ap:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=['No Default', 'Default']))

# %% [markdown]
# ## Step 6: Hyperparameter Optimization 🎯
# 
# Now let's optimize our model using systematic hyperparameter tuning.

# %%
# Define parameter grid for optimization
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

# Use a smaller grid for demonstration (full grid would take too long)
param_grid_small = {
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8],
    'gamma': [0, 1]
}

# Grid search
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False
)

grid_search = GridSearchCV(
    xgb_classifier,
    param_grid_small,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("🔍 Performing hyperparameter search...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"📈 Best cross-validation score: {grid_search.best_score_:.3f}")

# %% [markdown]
# ## Step 7: Final Model Training with Advanced Features 🏆

# %%
# Train final model with best parameters and advanced features
final_model = xgb.XGBClassifier(
    **grid_search.best_params_,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# Use early stopping
eval_set = [(X_train, y_train), (X_val, y_val)]
final_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=20,
    verbose=False
)

# Get predictions
y_pred_final = final_model.predict(X_test)
y_pred_proba_final = final_model.predict_proba(X_test)[:, 1]

# Calculate metrics
final_auc = roc_auc_score(y_test, y_pred_proba_final)
final_ap = average_precision_score(y_test, y_pred_proba_final)

print("🏆 Final Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.3f}")
print(f"ROC AUC: {final_auc:.3f} (Improvement: {(final_auc - baseline_auc) / baseline_auc * 100:.1f}%)")
print(f"Average Precision: {final_ap:.3f}")

# %% [markdown]
# ## Step 8: Model Interpretation and Business Insights 💡

# %%
# Comprehensive model evaluation
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[0, 0].barh(feature_importance['feature'][::-1], feature_importance['importance'][::-1])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 15 Most Important Features')

# 2. Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xticklabels(['No Default', 'Default'])
axes[0, 1].set_yticklabels(['No Default', 'Default'])

# 3. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_final)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True, alpha=0.3)

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_final)
axes[1, 1].plot(recall, precision, color='darkgreen', lw=2, label=f'AP = {final_ap:.3f}')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 9: Business Application - Risk Scoring 💼

# %%
# Create a risk scoring system
# Define risk categories based on predicted probabilities
risk_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Apply to test set
test_results = pd.DataFrame({
    'true_default': y_test,
    'predicted_probability': y_pred_proba_final,
    'risk_category': pd.cut(y_pred_proba_final, bins=risk_thresholds, labels=risk_labels)
})

# Analyze risk categories
risk_analysis = test_results.groupby('risk_category').agg({
    'true_default': ['count', 'sum', 'mean']
}).round(3)
risk_analysis.columns = ['Count', 'Defaults', 'Default_Rate']

print("📊 Risk Category Analysis:")
print(risk_analysis)

# Visualize risk distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Risk category distribution
test_results['risk_category'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Distribution of Customers by Risk Category')
ax1.set_xlabel('Risk Category')
ax1.set_ylabel('Number of Customers')

# Default rate by risk category
risk_analysis['Default_Rate'].plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Actual Default Rate by Risk Category')
ax2.set_xlabel('Risk Category')
ax2.set_ylabel('Default Rate')
ax2.set_ylim(0, 1)

# Add value labels on bars
for i, v in enumerate(risk_analysis['Default_Rate']):
    ax2.text(i, v + 0.01, f'{v:.1%}', ha='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 10: Making Business Decisions 📈
# 
# Let's simulate how the bank would use this model to make lending decisions.

# %%
# Business simulation
# Calculate potential profit/loss for different threshold strategies

# Business parameters
loan_profit_rate = 0.05  # 5% profit on successful loans
default_loss_rate = 0.30  # 30% loss on defaulted loans

# Test different approval thresholds
thresholds = np.arange(0.1, 0.7, 0.05)
business_results = []

for threshold in thresholds:
    # Approve loans below threshold
    approved = y_pred_proba_final < threshold
    
    # Calculate business metrics
    n_approved = approved.sum()
    n_defaults = ((y_test == 1) & approved).sum()
    n_successful = n_approved - n_defaults
    
    # Calculate profit/loss
    profit = n_successful * loan_profit_rate - n_defaults * default_loss_rate
    approval_rate = n_approved / len(y_test)
    default_rate_approved = n_defaults / n_approved if n_approved > 0 else 0
    
    business_results.append({
        'threshold': threshold,
        'approval_rate': approval_rate,
        'default_rate': default_rate_approved,
        'profit_per_100_loans': profit / len(y_test) * 100
    })

business_df = pd.DataFrame(business_results)

# Find optimal threshold
optimal_idx = business_df['profit_per_100_loans'].idxmax()
optimal_threshold = business_df.loc[optimal_idx, 'threshold']

# Visualize business impact
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Approval rate vs default rate
ax1.plot(business_df['threshold'], business_df['approval_rate'], 'b-', label='Approval Rate')
ax1.plot(business_df['threshold'], business_df['default_rate'], 'r-', label='Default Rate (Approved)')
ax1.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
ax1.set_xlabel('Risk Threshold')
ax1.set_ylabel('Rate')
ax1.set_title('Approval and Default Rates by Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Profit curve
ax2.plot(business_df['threshold'], business_df['profit_per_100_loans'], 'g-', linewidth=2)
ax2.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Risk Threshold')
ax2.set_ylabel('Profit per 100 Loans')
ax2.set_title('Expected Profit by Risk Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"💰 Optimal Strategy:")
print(f"Risk threshold: {optimal_threshold:.2f}")
print(f"Approval rate: {business_df.loc[optimal_idx, 'approval_rate']:.1%}")
print(f"Default rate among approved: {business_df.loc[optimal_idx, 'default_rate']:.1%}")
print(f"Expected profit per 100 loans: ${business_df.loc[optimal_idx, 'profit_per_100_loans']:.2f}")

# %% [markdown]
# ## 🎉 Congratulations!
# 
# You've successfully completed a comprehensive XGBoost learning journey! Here's what you've learned:
# 
# ### Theory & Fundamentals:
# ✅ How gradient boosting works  
# ✅ XGBoost's improvements over traditional boosting  
# ✅ Key parameters and their effects  
# ✅ Regularization and overfitting prevention  
# 
# ### Practical Skills:
# ✅ Data preprocessing and feature engineering  
# ✅ Model training and hyperparameter tuning  
# ✅ Model evaluation with multiple metrics  
# ✅ Business application and decision-making  
# 
# ### Next Steps:
# 1. Try XGBoost on your own datasets
# 2. Experiment with different objective functions
# 3. Explore XGBoost's advanced features (custom objectives, callbacks)
# 4. Learn about SHAP values for model interpretation
# 5. Compare with LightGBM and CatBoost
# 
# Happy modeling! 🚀

# %% [markdown]
# ## 📚 Additional Exercises for Practice
# 
# 1. **Feature Selection**: Use XGBoost's feature importance to select top features and retrain
# 2. **Class Imbalance**: Experiment with `scale_pos_weight` parameter
# 3. **Custom Metrics**: Implement a custom evaluation metric for business cost
# 4. **Ensemble Methods**: Combine XGBoost with other models
# 5. **Time Series**: Apply XGBoost to time-series prediction

# %%
# Save the model for deployment
import joblib

# Save the model
joblib.dump(final_model, 'credit_risk_xgboost_model.pkl')
print("✅ Model saved successfully!")

# Example: Loading and using the saved model
# loaded_model = joblib.load('credit_risk_xgboost_model.pkl')
# new_predictions = loaded_model.predict(new_data)