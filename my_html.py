from flask import Flask,url_for,render_template,request

app = Flask(__name__, template_folder='template')

@app.route('/para/<user>')
def index(user):
    return render_template('abc.html', user_template=user)

@app.route('/login', methods=['GET', 'POST']) 
def login():
    if request.method == 'POST': 
        return 'Hello ' + request.values['username'] 

    return "<form method='post' action='/login'><input type='text' name='username' />" \
            "</br>" \
        "<button type='submit'>Submit</button></form>"

if __name__ == '__main__':
    app.debug = True
    app.run()