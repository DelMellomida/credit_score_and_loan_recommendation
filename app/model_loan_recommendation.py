from flask import session
import sqlite3
import os

def get_loan_recommendation():
    # Retrieve values from the session
    credit_score = float(session.get('credit_score', 0))
    monthly_income = float(session.get('monthly_income', 0))
    monthly_debt_payment = float(session.get('monthly_debt_payment', 0))

    if not credit_score or not monthly_income or not monthly_debt_payment:
        return None, "Required data is missing."

    # Calculate the debt-to-income ratio
    debt_to_income_ratio = monthly_debt_payment / monthly_income if monthly_income > 0 else 0

    try:
        # Define the database path
        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, '../data', 'users.db')

        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Fetch loan options from the database
        cursor.execute("SELECT * FROM loan_options")
        rows = cursor.fetchall()

        loan_options = []
        for row in rows:
            loan = {
                'loan_id': row[0],
                'loan_name': row[1],
                'min_income': row[2],
                'max_dti': row[3],
                'interest_rate': row[4],
                'max_term': row[5],
                'min_credit_score': row[6],
                'notes': row[7]
            }
            loan_options.append(loan)

        # Check eligibility of loans based on the provided conditions
        eligible_loans = []
        for loan in loan_options:
            if (
                monthly_income >= loan['min_income'] and 
                debt_to_income_ratio <= loan['max_dti'] and 
                credit_score >= loan['min_credit_score']
            ):
                eligible_loans.append(loan)

        if not eligible_loans:
            return None, "No loan recommendations available based on the provided data."

        return eligible_loans, None

    except sqlite3.Error as e:
        return None, f"Database error: {e}"

    finally:
        connection.close()
