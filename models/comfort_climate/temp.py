import joblib

# Load the object from the file
data = joblib.load('/Users/aby/UTS/AMLA - Sem 4/Assign 2/sydney-weather-intelligence/models/comfort_climate/cci_ridge_h1.joblib')

# Now you can use it like a regular Python object
print(type(data))
print(data)
