from sklearn.externals import joblib
from flask import request, jsonify, Flask
import json
app = Flask(__name__)
@app.route('/', methods=['POST'])
def predict():
	if request.method == 'POST':
		try:
			data = request.get_json()
			sms = str(data['m'])
			sms=[sms]
			model = joblib.load('./clf.pkl')
			cv = joblib.load('./cv.pkl')
			sms  = cv.transform(sms).toarray()
		except ValueError:
			return jsonify('Please enter a number\t')
	return jsonify(model.predict(sms).tolist())

if __name__ == '__main__':
	app.run(debug=True)