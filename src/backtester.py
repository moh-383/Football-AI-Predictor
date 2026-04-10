def backtest(df, model, features_df, n_test_matches=200):
    """
    Predit les N derniers matchs du dataset (dont on connait le resultat)
    et calcule la precision reelle du modele.
    """
    test_set = features_df.tail(n_test_matches)
    X_test = test_set[FEATURE_COLS].values
    y_true = test_set["target"].values
    y_pred = model.predict(X_test)
 
    accuracy = (y_pred == y_true).mean()
    print(f"Accuracy sur {n_test_matches} matchs recents : {accuracy:.1%}")
 
    # Afficher les matchs ou le modele s'est le plus trompe
    # -> identifier les types de matchs difficiles a predire
