from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data
    
    """
    y_pred = model.predict(X_test)
    print(f'Classification report:\n {classification_report(y_test, y_pred)}')
    print(f'\nClassification matrix:\n {confusion_matrix(y_test, y_pred)}')