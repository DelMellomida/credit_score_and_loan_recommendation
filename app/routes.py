from flask import render_template

def init_routes(app):
    @app.route("/")
    def home():
        return render_template("index.html")  # Serves the HTML file from the templates folder
