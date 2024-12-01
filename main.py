from flask import Flask,render_template,request
import os
from Agent import MLAagent
app=Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')
@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        file = request.files['file']  # Correct way to access the file
        description = request.form['description']  # Text input remains in request.form
        file_path = file.filename
        file.save(file_path)
        MLAagent(problem_statement=description,dataset_path=file_path)

        
        # Add any processing logic here
    return render_template('project_updates.html')


if __name__=='__main__':
    app.run(debug=True)