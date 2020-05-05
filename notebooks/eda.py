# %%
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pdp
# %%
train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/test.csv')

# %%
print(train.head())
print(train.info())
print(train.isnull().sum())
print(train.head(5))
# %%
print(train.columns)
print(train.shape)
# %%
print(test.head())
print(test.info())
print(test.isnull().sum())


# %%
print(test.columns)
print(test.shape)

# %%
train_profile = pdp.ProfileReport(train)
# %%
train_profile.to_file("../features/importances/train_profile.html")


# %%
print(train['previous_cancellations'].describe())