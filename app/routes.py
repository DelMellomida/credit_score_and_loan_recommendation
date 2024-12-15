from flask import render_template, g
import joblib

def init_routes(app):
    @app.route("/")
    def home():
        return render_template("index.html")  # Serves the HTML file from the templates folder

def setup_routes(app):
    request = g.request
    results = g.results

    # Home Route
    @app.route('/')
    def index():
        return render_template('index.html') 

    @app.route('/form')
    def form():
        return render_template('predict_form.html') 

    @app.route('/predict', methods=['POST'])
    def predict():

        classifier = joblib.load('../saved_models/f1_Classifier_CreditScoring.pkl')
        scaler = joblib.load('../saved_models/scaler_CreditScoring.pkl')
        ohe = joblib.load('../saved_models/encoder_CreditScoring.pkl')

        try:
            # Extract input features from form
            features = [
                float(request.form['DerogCnt']),
                float(request.form['CollectCnt']),
                float(request.form['BanruptcyInd']),
                float(request.form['InqCnt06']),
                float(request.form['InqTimeLast']),
                float(request.form['InqFinanceCnt24']),
                float(request.form['TLTimeFirst']),
                float(request.form['TLTimeLast']),
                float(request.form['TLCnt03']),
                float(request.form['TLCnt12']),
                float(request.form['TLCnt24']),
                float(request.form['TLCnt']),
                float(request.form['TLSum'].replace('$', '').replace(',', '')),
                float(request.form['TLMaxSum'].replace('$', '').replace(',', '')),
                float(request.form['TLSatCnt']),
                float(request.form['TLDel60Cnt']),
                float(request.form['TLBadCnt24']),
                float(request.form['TL75UtilCnt']),
                float(request.form['TL50UtilCnt']),
                float(request.form['TLBalHCPct'].replace('%', '').replace(',', '')) / 100,
                float(request.form['TLSatPct'].replace('%', '').replace(',', '')) / 100,
                float(request.form['TLDel3060Cnt24']),
                float(request.form['TLDel90Cnt24']),
                float(request.form['TLDel60CntAll']),
                float(request.form['TLOpenPct'].replace('%', '').replace(',', '')) / 100,
                float(request.form['TLBadDerogCnt']),
                float(request.form['TLDel60Cnt24']),
                float(request.form['TLOpen24Pct'].replace('%', '').replace(',', '')) / 100
            ]
            
            # Convert to numpy array and reshape for the model
            input_data = np.array(features).reshape(1, -1)

            # Standardize and encode the input data (adjust accordingly)
            input_data_continuous = scaler.transform(input_data[:, :len(continuous_columns)])
            input_data_categorical = ohe.transform(input_data[:, len(continuous_columns):]).toarray()
            input_data_transformed = np.hstack([input_data_continuous, input_data_categorical])

            # Make prediction
            prediction = classifier.predict(input_data_transformed)
            prediction_probability = classifier.predict_proba(input_data_transformed)[0][1]

            return render_template(
                'results.html',
                prediction=int(prediction[0]),
                probability=round(prediction_probability * 100, 2)
            )
        except Exception as e:
            return f"An error occurred: {e}"


    # Metric Route
    @app.route('/metric')
    def metric():
        return render_template('metric.html', results=results)

    @app.route('/login')
    def login():
        return render_template('login.html')

    @app.route('/signup')
    def signup():
        return render_template('signup.html')
