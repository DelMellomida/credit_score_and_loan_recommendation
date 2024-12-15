import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
                static_folder=os.path.join(os.path.dirname(__file__), '../static'))

    # print("Template Folder Path:", app.template_folder)

    base_dir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, '../data', 'users.db')

    print(db_path)

    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = 'your_secret_key'

    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app


