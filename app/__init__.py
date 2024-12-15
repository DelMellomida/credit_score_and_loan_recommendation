import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # Get the absolute path of the 'data' folder
    base_dir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, 'data', 'users.db')

    # Set the correct database URI using the absolute path
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = 'your_secret_key'  # Replace with a secure key

    db.init_app(app)

    with app.app_context():
        db.create_all()  # Ensure the database tables are created

    from .routes import setup_routes
    setup_routes(app)

    return app
