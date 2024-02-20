from sklearn.linear_model import LinearRegression


# Create X and y arrays
X = sales_df.drop("sales", axis=1).Values
y = sales_df["sales"].Values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
y_pred = reg.predict(X)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))