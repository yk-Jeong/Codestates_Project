from flask import Flask, render_template, request
import model
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
# load the pickled model

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('similarity.html')


@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():

    import psycopg2
    conn = psycopg2.connect(
    host="castor.db.elephantsql.com",
    database="mzwyjwso",
    user="mzwyjwso",
    password="hYJRMpv0-HmiV_sfIUGRbxVnVg_wi7zk")

    return render_template('recommendation.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    name = request.form["input"]
    predict = model(name)

    return render_template('similarity_answer.html', data = predict)



@app.route('/feedback', methods = ['GET', 'POST'])
def feedback():

    import psycopg2
    conn = psycopg2.connect(
    host="drona.db.elephantsql.com",
    database="ehlxypld",
    user="ehlxypld",
    password="xsfHBK5g3aN2rm8aXn86FZRnpj752BsI")    

    return render_template('feedback.html')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

if __name__ == "__main__":
    app.run(debug=True)
