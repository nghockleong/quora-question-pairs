from sklearn.model_selection import RandomizedSearchCV

def tune_best_params(base_model, param_grid, X, y, num_iter, scoring_metric, cv_obj):
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions = param_grid,
        n_iter = num_iter,
        cv = cv_obj,
        scoring = scoring_metric,
        random_state = 42
    )
    random_search.fit(X, y)
    return random_search

def print_train_test_accuracy_comparison(
        best_model
        , model_name
        , X_train
        , y_train
        , X_test
        , y_test):
    best_model.fit(X_train, y_train)
    print(f"Train accuracy of {model_name}: ", best_model.score(X_train, y_train))
    print(f"Test accuracy of {model_name}: ", best_model.score(X_test, y_test))
    return best_model