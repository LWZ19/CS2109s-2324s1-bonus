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
    df_entropy = cond_column.apply(lambda x: entropy(x.value_counts(), base=base))

    df_sum_rows = df[c_col].value_counts(normalize=True)
    return (df_entropy.sort_index() * df_sum_rows.sort_index()).sum()


def info_gain(df, Y, X, base=2):
    return pd_entropy(df, Y, base) - pd_entropy_c(df, Y, X, base)


def create_tree(tree, df, parent=None, action=''):

    info_gain_dicts = []
    best_col = None
    best_ig = -np.inf
    decision = df.columns[-1]

    # Check for Same classification
    decision_values = df[decision].value_counts()
    if decision_values.shape[0] == 1:
        label = '{}{}:{}'.format(action, decision_values.keys()[0], decision_values[0])
        tree.create_node(label, best_col, parent=parent)
        return

    # Check if the data is empty
    if df.shape[1] == 1:
        deci_values = df[decision].value_counts()
        deci_keys = deci_values.keys()
        result = action
        for i in range(len(deci_keys)):
            result += '{}:{} '.format(deci_keys[i], deci_values[i])

        tree.create_node(result, best_col, parent=parent)
        return

    # Code to determine the best col, remember to skip Decision column.
    for col in df.columns[:-1]:
        curr_info_gain = info_gain(df, decision, col)
        if curr_info_gain > best_ig:
            best_ig = curr_info_gain
            best_col = col

    tree.create_node(action + best_col, best_col, parent=parent)

    # Code to get the next branch of the tree
    keys = df[best_col].value_counts().keys()
    for key in keys:
        new_df = df[df[best_col] == key].drop(best_col, axis=1)
        new_action = '[{}] '.format(key)
        create_tree(tree, new_df, parent=best_col, action=new_action)


tree = Tree()
create_tree(tree, loan_df)
#tree.save2file('tree.txt', line_type='ascii')
tree.show(line_type='ascii')

loan_noisy_df = loan_df.copy()
loan_noisy_df.iloc[0]['Decision'] = 'Approve'

tree = Tree()
create_tree(tree, loan_noisy_df)
#tree.save2file('tree.txt', line_type='ascii')
tree.show(line_type='ascii')
