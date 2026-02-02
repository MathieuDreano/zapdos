import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import train_test_split

# -----------------------------
# Prepare dataset
# -----------------------------
np.random.seed(42)

data = pd.read_csv("products.csv")
categorical_cols = ['brand', 'sports', 'productnature', 'structurationvalues']
X = data.drop(columns=["category_id"])
for col in categorical_cols:
    X[col] = X[col].astype('category')

# -----------------------------
# Train-test split
# -----------------------------

y = data["category_id"]
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.33,random_state=42)
# Keep only test rows whose labels exist in training set
y_train_unique = set(y_train)
mask = y_test.isin(y_train_unique)

x_test_filtered = x_test[mask]
y_test_filtered = y_test[mask]

# -----------------------------
# Train LightGBM
# -----------------------------
print("Training LightGBM model...")

model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1, random_state=42)

model.fit(
    x_train,
    y_train,  # use original labels
    eval_set=[(x_test_filtered, y_test_filtered)],
    eval_metric='logloss',
    categorical_feature=categorical_cols
)
# -----------------------------
# 3. Extract tree-based structure
# -----------------------------
# Each product ends up in a leaf per tree
leaf_indices = model.predict(X, pred_leaf=True)

# Build hierarchical grouping
category_tree = defaultdict(lambda: defaultdict(list))

for idx, leaves in enumerate(leaf_indices):
    top_node = f"leaf_group_{leaves[0]}"
    sub_node = f"leaf_group_{leaves[1]}"
    category_tree[top_node][sub_node].append(idx)

# -----------------------------
# 4. Pretty print tree
# -----------------------------
def print_tree(tree, depth=0):
    for node, children in tree.items():
        print("  " * depth + f"- {node}")
        if isinstance(children, dict):
            print_tree(children, depth + 1)
        else:
            print("  " * (depth + 1) + f"{len(children)} products")

print("\nGenerated Category Tree:")
print_tree(category_tree)

# -----------------------------
# 5. Analyze categories per leaf
# -----------------------------

# Pick one tree (e.g., first)
tree_leaves = leaf_indices[:, 0]

# Add leaf indices to your dataset
X['leaf_id'] = tree_leaves
X['category_id'] = y.values

# Summarize common attributes per leaf
for leaf in X['leaf_id'].unique():
    subset = X[X['leaf_id'] == leaf]
    print(f"Leaf {leaf} - {len(subset)} products")
    for col in categorical_cols:
        print(f"  {col} most common:", subset[col].value_counts().head(3).to_dict())

