from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle as pk

app = Flask(__name__)
model = pk.load(open('model.pkl', 'rb'))


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    inp = [float(x) for x in request.form.values()]
    inp = [np.array(inp)]
    fi = sc.fit_transform(inp)
    res = model.predict(fi)

    return render_template('index.html', pred_out="This asteroid might be {}".format("Hazardous" if res[0] == 'True' else "not Hazardous"))


if __name__ == '__main__':
    app.run(debug=True)