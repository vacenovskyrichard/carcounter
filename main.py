import os
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import Flask, session
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/vacenric/carcounter_website/uploads'
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('home_page.html')


@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/result')
def detector():
    return render_template('result.html')


if __name__ == "__main__":
    app.run(debug=True)
