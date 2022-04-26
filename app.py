import pandas as pd
from flask import Flask, jsonify, request
import pickle
# Code from Best Pipeline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.46883208314205954
exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=7, max_features=0.35000000000000003, min_samples_leaf=17, min_samples_split=20, n_estimators=100, subsample=0.8500000000000001)

exported_pipeline.fit(training_features, training_target)


# Flask app script
#app
app = Flask(__name__)
#routes
@app.route('/', methods=['POST'])
def predict():
    #get data
    
    data = request.get_json(force=True)
    
    #convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    #predictions
    
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    try:
        data_df = encoder.transform(data_df)
    except:
      print("Encoder transform failed")
            
    result = exported_pipeline.predict(data_df)
   
    #send back to browser
    output = {'results': result[0]}
    
    #return data
    return jsonify(results=output)
    
if __name__ == "__main__":
    # app.run(debug = True)
    app.run(host ='0.0.0.0', port = 8080, debug = True)
    
