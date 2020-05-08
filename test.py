# %%
from utils import load_datasets, load_target
import argparse
import json

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))
# %%
feats = config['features']
_, X_test = load_datasets(feats)
# %%
print(X_test.shape)
