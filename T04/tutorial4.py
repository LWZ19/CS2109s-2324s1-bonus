import pandas as pd
import numpy as np
from scipy.stats import entropy
from treelib import Tree

# Question 1
loan_df = pd.DataFrame(np.array([
    ['Over 10k', 'Bad', 'Low', 'Reject'],
    ['Over 10k', 'Good', 'High', 'Approve'],
    ['0 - 10k', 'Good', 'Low', 'Approve'],
    ['Over 10k', 'Good', 'Low', 'Approve'],
    ['Over 10k', 'Good', 'Low', 'Approve'],
    ['Over 10k', 'Good', 'Low', 'Approve'],
    ['0 - 10k', 'Good', 'Low', 'Approve'],
    ['Over 10k', 'Bad', 'Low', 'Reject'],
    ['Over 10k', 'Good', 'High', 'Approve'],
    ['0 - 10k', 'Bad', 'High', 'Reject'],
]), columns=['Income', 'Credit History', 'Debt', 'Decision'])

print(loan_df.to_markdown())

def pd_entropy(df, col, base=2):
    return entropy(pd.Series(df[col]).value_counts(normalize=True, sort=False), base=base)

def pd_entropy_c(df, col, c_col, base=2):
    cond_column = df.groupby(c_col)[col]
    df_entropy = cond_column.apply(lambda x:entropy(x.value_counts(), base=base))

    df_sum_rows = df[c_col].value_counts(normalize=True)
    return (df_entropy.sort_index()*df_sum_rows.sort_index()).sum()

def info_gain(df, Y, X, base=2):
    return pd_entropy(df, Y, base) - pd_entropy_c(df, Y, X, base)

def create_tree(tree, df, parent=None, action = ''):

    info_gain_dicts = []
    best_col = None
    best_ig = -np.inf

    # Code to determine the best col, remember to skip Decision column.

    tree.create_node(action + best_col, best_col, parent=parent)
    
    # Code to get the next branch of the tree

tree = Tree()
create_tree(tree, loan_df)
tree.save2file('tree.txt', line_type='ascii')

loan_noisy_df = loan_df.copy()
loan_noisy_df.iloc[0]['Decision'] = 'Approve'

tree = Tree()
create_tree(tree, loan_noisy_df)
tree.save2file('tree.txt', line_type='ascii')