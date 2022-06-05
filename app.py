from flask import Flask , render_template,request
import pickle
import numpy as np

model = pickle.load(open('model_plk','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)

	if prediction ==  1:
		output = "patienet has diabetes"
	else:
		output = "patienet doesn't have diabetes"

	#output = prediction(,5)

	return render_template('index.html',prediction_text='{}'.format(output))
	

if __name__ == "__main__":
	app.run()




