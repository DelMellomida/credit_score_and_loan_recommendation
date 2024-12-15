from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from app import create_app
from werkzeug.security import generate_password_hash, check_password_hash
from app.user_model import User
from app.user_financial_model import UserFinancial
from app import db
from app.model_loan_recommendation import get_loan_recommendation
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = create_app()

# Load the model and preprocessor
classifier = joblib.load('saved_models/f1_Classifier_CreditScoring.pkl')
scaler = joblib.load('saved_models/scaler_CreditScoring.pkl')
ohe = joblib.load('saved_models/encoder_CreditScoring.pkl')

# Load dataset and preprocess (similar to your code)
df = pd.read_excel(r"data/a_Dataset_CreditScoring.xlsx")
df = df.drop('ID', axis=1)

# Step 1: Discard rows with more than 50% missing values
row_threshold = len(df.columns) / 2
df = df.dropna(thresh=row_threshold, axis=0)

# Step 2: Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical variables
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Continuous variables
        df[col].fillna(df[col].mean(), inplace=True)

# Step 3: Discard columns with more than 95% missing values
column_threshold = 0.95 * len(df)
df = df.loc[:, df.isnull().sum() < column_threshold]

# Step 4: Define target and features
y = df['TARGET']
X = df.drop(columns=['TARGET'])

# Step 5: Standardization and encoding
categorical_columns = X.select_dtypes(include=['object']).columns
continuous_columns = X.select_dtypes(exclude=['object']).columns

# Encode categorical features
X_categorical = ohe.transform(X[categorical_columns]).toarray()
X_continuous = scaler.transform(X[continuous_columns])

# Combine features
X_transformed = np.hstack([X_continuous, X_categorical])

# Predict using the trained model
y_pred = classifier.predict(X_transformed)

# Step 5: Compute metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

# Step 6: Prepare results to send to HTML
results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'conf_matrix': conf_matrix,
    'class_report': class_report
}

def compute_credit_score(features):
    # Initialize the credit score, typically set to 700 as a baseline.
    credit_score = 700

    # 1. Penalize for high number of derogatory marks (DerogCnt)
    if features['DerogCnt'] > 0:
        credit_score -= features['DerogCnt'] * 30  # Penalize 30 points for each derogatory mark
        # Max penalty for derogatory marks
        if credit_score < 300:
            credit_score = 300
    
    # 2. Penalize for number of collection accounts (CollectCnt)
    if features['CollectCnt'] > 0:
        credit_score -= features['CollectCnt'] * 20  # Penalize 20 points for each collection account
        if credit_score < 300:
            credit_score = 300

    # 3. Penalize for bankruptcy (BanruptcyInd)
    if features['BanruptcyInd'] == 1:
        credit_score -= 100  # Severe penalty for bankruptcy
        if credit_score < 300:
            credit_score = 300

    # 4. Adjust for credit inquiries (InqCnt06 and InqTimeLast)
    if features['InqCnt06'] > 0:
        credit_score -= features['InqCnt06'] * 10  # Penalize for each inquiry in the last 6 months
    if features['InqTimeLast'] < 6:
        credit_score -= 30  # Penalize if the last inquiry was within the past 6 months

    # 5. Adjust for financing inquiries (InqFinanceCnt24)
    if features['InqFinanceCnt24'] > 0:
        credit_score -= features['InqFinanceCnt24'] * 15  # Penalize for financing inquiries in the last 24 months

    # 6. Adjust for the length of time the first credit account has been open (TLTimeFirst)
    if features['TLTimeFirst'] < 24:  # Penalize if the account is less than 2 years old
        credit_score -= 20

    # 7. Adjust for the length of time the last credit account has been opened (TLTimeLast)
    if features['TLTimeLast'] < 6:  # Penalize if the last account opened is within the last 6 months
        credit_score -= 15

    # 8. Number of accounts with delinquencies (TLCnt03, TLCnt12, TLCnt24)
    delinquency_penalty = (features['TLCnt03'] * 10) + (features['TLCnt12'] * 20) + (features['TLCnt24'] * 30)
    credit_score -= delinquency_penalty

    # 9. Penalize for the total number of credit accounts (TLCnt) if it's too high
    if features['TLCnt'] > 10:
        credit_score -= (features['TLCnt'] - 10) * 5  # Penalize 5 points for each account over 10

    # 10. Add points for total loan sum (TLSum)
    if features['TLSum'] > 5000:
        credit_score += 20  # Reward for higher loan sums

    # 11. Add points for maximum loan sum (TLMaxSum)
    if features['TLMaxSum'] > 10000:
        credit_score += 30  # Reward for larger loan sums

    # 12. Positive points for satisfactory accounts (TLSatPct)
    credit_score += features['TLSatPct'] * 100  # Max weight on satisfactory accounts

    # 13. Penalize if accounts have over-utilized credit (TL75UtilCnt, TL50UtilCnt)
    credit_score -= features['TL75UtilCnt'] * 25  # Penalize for accounts with over 75% utilization
    credit_score -= features['TL50UtilCnt'] * 15  # Penalize for accounts with over 50% utilization

    # 14. Consider high credit balance percentage (TLBalHCPct)
    if features['TLBalHCPct'] > 1:
        credit_score -= (features['TLBalHCPct'] - 1) * 50  # Penalize for over-utilization

    # 15. Adjust for accounts with serious delinquencies (TLBadCnt24)
    credit_score -= features['TLBadCnt24'] * 50  # Penalize for each account with serious delinquencies

    # 16. Reward for open accounts (TLOpenPct, TLOpen24Pct)
    credit_score += features['TLOpenPct'] * 50  # Reward for high percentage of open accounts
    credit_score += features['TLOpen24Pct'] * 50  # Reward for high percentage of open accounts in last 24 months

    # Ensure the score is within the acceptable range (300-850)
    credit_score = max(300, min(credit_score, 850))

    return credit_score

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('dashboard.html')
    return render_template('index.html')

@app.route('/form')
def form():
    if 'user_id' not in session:
        flash("You must be logged in to access the prediction form.", 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    financial_info = UserFinancial.query.filter_by(user_id=user_id).first()

    if not financial_info or financial_info.monthly_income == 0 or financial_info.monthly_debt_payment == 0:
        flash("Please fill out your financial information before accessing the prediction form.", 'warning')
        return redirect(url_for('basicForm'))

    return render_template('predict_form.html') 


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash("You must be logged in to access the dashboard.", 'danger')
        return redirect(url_for('login'))

    try:
        features = {
            'DerogCnt': float(request.form['DerogCnt']),
            'CollectCnt': float(request.form['CollectCnt']),
            'BanruptcyInd': float(request.form['BanruptcyInd']),
            'InqCnt06': float(request.form['InqCnt06']),
            'InqTimeLast': float(request.form['InqTimeLast']),
            'InqFinanceCnt24': float(request.form['InqFinanceCnt24']),
            'TLTimeFirst': float(request.form['TLTimeFirst']),
            'TLTimeLast': float(request.form['TLTimeLast']),
            'TLCnt03': float(request.form['TLCnt03']),
            'TLCnt12': float(request.form['TLCnt12']),
            'TLCnt24': float(request.form['TLCnt24']),
            'TLCnt': float(request.form['TLCnt']),
            'TLSum': float(request.form['TLSum'].replace('$', '').replace(',', '')),
            'TLMaxSum': float(request.form['TLMaxSum'].replace('$', '').replace(',', '')),
            'TLSatCnt': float(request.form['TLSatCnt']),
            'TLDel60Cnt': float(request.form['TLDel60Cnt']),
            'TLBadCnt24': float(request.form['TLBadCnt24']),
            'TL75UtilCnt': float(request.form['TL75UtilCnt']),
            'TL50UtilCnt': float(request.form['TL50UtilCnt']),
            'TLBalHCPct': float(request.form['TLBalHCPct'].replace('%', '').replace(',', '')) / 100,
            'TLSatPct': float(request.form['TLSatPct'].replace('%', '').replace(',', '')) / 100,
            'TLDel3060Cnt24': float(request.form['TLDel3060Cnt24']),
            'TLDel90Cnt24': float(request.form['TLDel90Cnt24']),
            'TLDel60CntAll': float(request.form['TLDel60CntAll']),
            'TLOpenPct': float(request.form['TLOpenPct'].replace('%', '').replace(',', '')) / 100,
            'TLBadDerogCnt': float(request.form['TLBadDerogCnt']),
            'TLDel60Cnt24': float(request.form['TLDel60Cnt24']),
            'TLOpen24Pct': float(request.form['TLOpen24Pct'].replace('%', '').replace(',', '')) / 100
        }

        input_data = np.array(list(features.values())).reshape(1, -1)

        scaled_data = scaler.transform(input_data)

        prediction_probability = classifier.predict_proba(scaled_data)[0][1]

        credit_score = compute_credit_score(features)
        session['credit_score'] = credit_score

        prediction = 1 if prediction_probability > 0.5 else 0

        # Save the data to Excel with correct format
        prediction_data = {
            'TARGET': prediction,  # Set target as prediction
            'ID': secure_filename(str(session.get('user_id'))),  # Unique ID (user ID in this case)
            **features
        }

        # Convert features back to their original format if needed
        formatted_prediction_data = {
            'TARGET': prediction,
            'ID': secure_filename(str(session.get('user_id'))),
            'DerogCnt': int(features['DerogCnt']),
            'CollectCnt': int(features['CollectCnt']),
            'BanruptcyInd': int(features['BanruptcyInd']),
            'InqCnt06': int(features['InqCnt06']),
            'InqTimeLast': int(features['InqTimeLast']),
            'InqFinanceCnt24': int(features['InqFinanceCnt24']),
            'TLTimeFirst': int(features['TLTimeFirst']),
            'TLTimeLast': int(features['TLTimeLast']),
            'TLCnt03': int(features['TLCnt03']),
            'TLCnt12': int(features['TLCnt12']),
            'TLCnt24': int(features['TLCnt24']),
            'TLCnt': int(features['TLCnt']),
            'TLSum': f"${features['TLSum']:,.2f}",  # Format as currency
            'TLMaxSum': f"${features['TLMaxSum']:,.2f}",  # Format as currency
            'TLSatCnt': int(features['TLSatCnt']),
            'TLDel60Cnt': int(features['TLDel60Cnt']),
            'TLBadCnt24': int(features['TLBadCnt24']),
            'TL75UtilCnt': int(features['TL75UtilCnt']),
            'TL50UtilCnt': int(features['TL50UtilCnt']),
            'TLBalHCPct': f"{features['TLBalHCPct']*100:.2f}%",  # Format as percentage
            'TLSatPct': f"{features['TLSatPct']*100:.2f}%",  # Format as percentage
            'TLDel3060Cnt24': int(features['TLDel3060Cnt24']),
            'TLDel90Cnt24': int(features['TLDel90Cnt24']),
            'TLDel60CntAll': int(features['TLDel60CntAll']),
            'TLOpenPct': f"{features['TLOpenPct']*100:.2f}%",  # Format as percentage
            'TLBadDerogCnt': int(features['TLBadDerogCnt']),
            'TLDel60Cnt24': int(features['TLDel60Cnt24']),
            'TLOpen24Pct': f"{features['TLOpen24Pct']*100:.2f}%"  # Format as percentage
        }

        formatted_prediction_df = pd.DataFrame([formatted_prediction_data])

        # Path to save the Excel file
        excel_file_path = os.path.join('data', 'f4_NewApplications_CreditScore_Predictions.xlsx')

        # Check if the file already exists
        if os.path.exists(excel_file_path):
            # Append to existing file
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                formatted_prediction_df.to_excel(writer, sheet_name='Predictions', index=False, header=False, startrow=writer.sheets['Predictions'].max_row)
        else:
            # Create a new file
            with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                formatted_prediction_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Initialize loan_recommendations to be an empty list
        loan_recommendations = []

        # Call get_loan_recommendation to get the loan recommendations
        loan_recommendations, error = get_loan_recommendation()

        # If there was an error in getting loan recommendations (e.g., no eligible loans),
        # loan_recommendations will be None, so you need to handle that case.
        if error:
            loan_recommendations = []

        # Calculate the DTI (Debt-to-Income ratio)
        monthly_income = float(session.get('monthly_income', 0))
        monthly_debt_payment = float(session.get('monthly_debt_payment', 0))

        dti = monthly_debt_payment / monthly_income if monthly_income > 0 else 0

        # Pass the DTI to the template
        return render_template(
            'results.html',
            prediction=prediction,
            probability=round(prediction_probability * 100, 2),
            credit_score=credit_score,
            loan_recommendations=loan_recommendations,
            dti=dti 
        )

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/metric')
def metric():
    return render_template('metric.html', results=results)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return render_template('dashboard.html')

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id

            user_financial = UserFinancial.query.filter_by(user_id=user.id).first()

            if not user_financial:
                user_financial = UserFinancial(user_id=user.id, monthly_income=0, monthly_debt_payment=0)
                db.session.add(user_financial)
                db.session.commit()

            session['monthly_income'] = user_financial.monthly_income
            session['monthly_debt_payment'] = user_financial.monthly_debt_payment

            flash("Login successful!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password.", 'danger')
            return redirect(url_for('login')) 

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return render_template('dashboard.html')
    
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        privacy_agreement = request.form.get('privacy-checkbox')

        if not privacy_agreement:
            flash('You must agree to the Privacy Policy and Terms of Service to sign up.')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match!", 'danger')
            return redirect(url_for('signup'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already exists!", 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(email=email, username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash("User created successfully!", 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/basic_form', methods=['GET', 'POST'])
def basicForm():
    if request.method == 'POST':
        user_id = session['user_id']
        occupation = request.form['occupation']
        monthly_income = request.form['monthly_income']
        monthly_debt_payment = request.form['monthly_debt_payment']

        financial_info = UserFinancial.query.filter_by(user_id=user_id).first()
        if financial_info:
            financial_info.monthly_income = monthly_income
            financial_info.monthly_debt_payment = monthly_debt_payment
            financial_info.occupation = occupation
        else:
            financial_info = UserFinancial(user_id=user_id, 
                                           monthly_income=monthly_income, 
                                           monthly_debt_payment=monthly_debt_payment,
                                           occupation=occupation)
            db.session.add(financial_info)

        session['monthly_income'] = monthly_income
        session['monthly_debt_payment'] = monthly_debt_payment

        db.session.commit()
        flash("Financial information updated successfully!", 'success')
        return redirect(url_for('form'))

    return render_template('basic_form.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("You must be logged in to access the dashboard.", 'danger')
        return redirect(url_for('login'))

    # Retrieve financial data from the session
    monthly_income = session.get('monthly_income')
    monthly_debt_payment = session.get('monthly_debt_payment')

    return render_template('dashboard.html', monthly_income=monthly_income, monthly_debt_payment=monthly_debt_payment)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.", 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
