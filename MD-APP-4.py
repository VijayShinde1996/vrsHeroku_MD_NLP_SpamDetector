from flask import Flask,render_template,url_for,request
import joblib

App4 = Flask(__name__)
# load the model from disk
clf = joblib.load('train_md4.pkl')
cv = joblib.load('transform.pkl')

@App4.route('/')
def home():
	return render_template('home.html')

@App4.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	App4.run(debug=True)