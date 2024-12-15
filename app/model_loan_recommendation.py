from flask import session

def get_loan_recommendation():
    # Retrieve values from the session
    credit_score = float(session.get('credit_score', 0))
    monthly_income = float(session.get('monthly_income', 0))
    monthly_debt_payment = float(session.get('monthly_debt_payment', 0))
    
    if not credit_score or not monthly_income or not monthly_debt_payment:
        return None, "Required data is missing."

    # Define the loan types and their requirements with more flexibility
    loan_options = [
        {
            'loan_id': 1,
            'loan_name': 'Personal Loan Basic',
            'min_income': 12000,  # Lowered minimum income
            'max_dti': 0.45,      # Increased DTI
            'interest_rate': 5.00,
            'max_term': 36,
            'min_credit_score': 550,  # Lowered minimum credit score
            'notes': 'Suitable for small needs'
        },
        {
            'loan_id': 2,
            'loan_name': 'Auto Loan Premium',
            'min_income': 20000,  # Lowered minimum income
            'max_dti': 0.40,      # Relaxed DTI condition
            'interest_rate': 4.00,  # Slightly reduced interest rate
            'max_term': 60,
            'min_credit_score': 620,  # Lowered minimum credit score
            'notes': 'Includes auto insurance'
        },
        {
            'loan_id': 3,
            'loan_name': 'Home Loan Standard',
            'min_income': 35000,  # Lowered minimum income
            'max_dti': 0.35,      # Kept DTI similar to before
            'interest_rate': 4.00,  # Slightly reduced interest rate
            'max_term': 240,
            'min_credit_score': 680,  # Lowered minimum credit score
            'notes': 'Long-term housing finance'
        },
        {
            'loan_id': 4,
            'loan_name': 'Quick Cash Advance',
            'min_income': 8000,  # Lowered minimum income
            'max_dti': 0.8,      # Increased DTI
            'interest_rate': 15.00,  # Higher interest rate
            'max_term': 12,
            'min_credit_score': 300,  # Lowered minimum credit score
            'notes': 'For emergencies only'
        },
        {
            'loan_id': 5,
            'loan_name': 'Debt Consolidation Loan',
            'min_income': 15000,  # Lowered minimum income for more flexibility
            'max_dti': 0.55,      # Increased DTI for more people to qualify
            'interest_rate': 7.00,  # Medium interest rate
            'max_term': 48,
            'min_credit_score': 580,  # Lowered minimum credit score
            'notes': 'Consolidate existing debts into one loan'
        },
        {
            'loan_id': 6,
            'loan_name': 'Small Business Loan',
            'min_income': 20000,  # Lowers the income for more applicants
            'max_dti': 0.50,      # Slightly relaxed DTI ratio
            'interest_rate': 8.00,  # Medium interest rate for business
            'max_term': 120,
            'min_credit_score': 650,  # Reasonable credit score for a business loan
            'notes': 'For small business owners'
        }
    ]

    # Calculate the debt-to-income ratio
    debt_to_income_ratio = monthly_debt_payment / monthly_income if monthly_income > 0 else 0
    
    # Find eligible loans based on income, DTI ratio, and credit score
    eligible_loans = []
    for loan in loan_options:
        if monthly_income >= loan['min_income'] and debt_to_income_ratio <= loan['max_dti'] and credit_score >= loan['min_credit_score']:
            eligible_loans.append(loan)
    
    if not eligible_loans:
        return None, "No loan recommendations available based on the provided data."

    return eligible_loans, None
