import os
import sqlite3
import numpy as np
import pickle

from flask import Flask, render_template, request
from flask_app.model import DATABASE_PATH

CSV_FILEPATH = os.path.join(os.getcwd(), __name__, 'student-mat.csv')

def create_app():
    app = Flask(__name__)
    #from flask_app.model import model_bp
    #app.register_blueprint(model_bp)
    model = pickle.load(open('flask_app/model/grade_model.pickle', 'rb'))
    
    @app.route('/')
    def home():
        return render_template('home.html')
    
    @app.route('/answer-me', methods=['GET', 'POST'])
    def survey():
        return render_template('survey.html')
    

    @app.route('/tada', methods=['POST'])
    def guess_grade():
        if request.method == 'POST':
            failures = request.form['failure']
            M_edu = request.form['mom']
            higher = request.form['higher']
            age = request.form['age']
            F_edu = request.form['pap']
            go_out = request.form['goout']
            romantic = request.form['romantic']
            travel = request.form['travel']
            
            survey = np.array([[age, failures, higher, romantic, M_edu, F_edu, go_out, travel]])
            pred = model.predict(survey)

            # 결과가 0점 미만으로 나올 경우 0으로 간주
            if pred[0] < 0:
                output = 0
            else:
                output = round(pred[0])

            # 입력값과 예측 결과가 db에 연동되도록 함
            
            conn = sqlite3.connect(DATABASE_PATH)
            cur = conn.cursor()

            cur.execute("INSERT INTO Info(age, failures, higher, romantic, M_edu, F_edu, go_out, travel, grade) VALUES('{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(age, failures, higher, romantic, M_edu, F_edu, go_out, travel, output))
            conn.commit()
            cur.close()
            

            return render_template('result.html', value=output)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)



