from . import db

class UserFinancial(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    monthly_income = db.Column(db.Numeric(10, 2), nullable=False)  # Store income as a decimal value
    monthly_debt_payment = db.Column(db.Numeric(10, 2), nullable=False)  # Store debt payment as a decimal value

def __repr__(self):
        return f'<User {self.username}>'