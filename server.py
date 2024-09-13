import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
from fuzzywuzzy import process

class NumpyInt64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

app = Flask(__name__)
CORS(app)
data = pd.read_csv('dataset.csv')
model = pickle.load(open('model.pkl', 'rb'))

def merge(dataset):
    grouped_data = data.groupby(['player_id', 'player'], as_index=False).first()
    merged_df = pd.merge(dataset, grouped_data[['player_id', 'player']], on='player_id', how='left')
    result_df = pd.concat([dataset, merged_df[['player']]], axis=1)
    print(result_df.columns)
    return result_df

@app.route('/getPlayer')
def getPlayer():
    name = request.args.get('name')

    best_match = process.extractOne(name, data['player'])
    best_match_index = best_match[2]

    matched_data = data.loc[best_match_index]
    matched_data = matched_data.fillna(value='NaN')
    matched_data_dict = matched_data.to_dict()
    
    for key, value in matched_data_dict.items():
        if isinstance(value, np.int64):
            matched_data_dict[key] = int(value)

    response_json = json.dumps({'player': matched_data_dict}, cls=NumpyInt64Encoder)

    return json.loads(response_json)

@app.route('/best')
def get_best():
    c_file = pd.read_csv('C.csv')
    c_file = merge(c_file)
    sg_file = pd.read_csv('SG.csv')
    sg_file = merge(sg_file)
    pg_file = pd.read_csv('PG.csv')
    pg_file = merge(pg_file)
    pf_file = pd.read_csv('PF.csv')
    pf_file = merge(pf_file)
    sf_file = pd.read_csv('SF.csv')
    sf_file = merge(sf_file)
    

    c_file = c_file.fillna(value='NaN')
    sg_file = sg_file.fillna(value='NaN')
    pg_file = pg_file.fillna(value='NaN')
    pf_file = pf_file.fillna(value='NaN')
    sf_file = sf_file.fillna(value='NaN')

    c = c_file.head(6).to_dict(orient='records')
    sg = sg_file.head(6).to_dict(orient='records')
    pg = pg_file.head(6).to_dict(orient='records')
    pf = pf_file.head(6).to_dict(orient='records')
    sf = sf_file.head(6).to_dict(orient='records')

    return jsonify(
        {'centers': c,
         'shootingGuards': sg,
         'pointGuards': pg,
         'powerForwards': pf,
         'smallForwards': sf,
         })

@app.route('/worst')
def get_worst():
    c_file = pd.read_csv('C.csv')
    c_file = merge(c_file)
    sg_file = pd.read_csv('SG.csv')
    sg_file = merge(sg_file)
    pg_file = pd.read_csv('PG.csv')
    pg_file = merge(pg_file)
    pf_file = pd.read_csv('PF.csv')
    pf_file = merge(pf_file)
    sf_file = pd.read_csv('SF.csv')
    sf_file = merge(sf_file)

    c_file = c_file.fillna(value='NaN')
    sg_file = sg_file.fillna(value='NaN')
    pg_file = pg_file.fillna(value='NaN')
    pf_file = pf_file.fillna(value='NaN')
    sf_file = sf_file.fillna(value='NaN')

    c = c_file.tail(5).to_dict(orient='records')
    sg = sg_file.tail(5).to_dict(orient='records')
    pg = pg_file.tail(5).to_dict(orient='records')
    pf = pf_file.tail(5).to_dict(orient='records')
    sf = sf_file.tail(5).to_dict(orient='records')

    return jsonify(
        {'centers': c,
         'shootingGuards': sg,
         'pointGuards': pg,
         'powerForwards': pf,
         'smallForwards': sf,
         })

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(v[1]) for v in request.form.items()]

    stat_cols = ['g', 'mp', 'fg', 'fga', 'x3p', 'x2p', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

    input_dict = dict(zip(stat_cols, input_values))

    input_df = pd.DataFrame([input_dict])

    missing_cols = set(stat_cols) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[stat_cols]

    input_df = np.squeeze(input_df, axis=0)
    print(input_df)
    prediction = model.predict([input_df])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
