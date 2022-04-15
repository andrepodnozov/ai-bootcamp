import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('ai-bootcamp/model1/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.46883208314205954
exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=7, max_features=0.35000000000000003, min_samples_leaf=17, min_samples_split=20, n_estimators=100, subsample=0.8500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
