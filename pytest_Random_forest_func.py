import pandas as pd
from sklearn.datasets import make_classification
from Clean_code import model_Random_Forest



def test_model_Random_Forest():#Vérifie que le retour comporte 10 var
    # Créer un jeu de données de test
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

    # Appeler la fonction
    feature_importances = model_Random_Forest(X, y)

    # Vérifier que le résultat est un DataFrame
    assert isinstance(feature_importances, pd.DataFrame)

    # Vérifier que le DataFrame a la bonne taille
    assert feature_importances.shape[0] == 10

    # Vous pouvez également ajouter d'autres assertions ici, comme vérifier le type des colonnes
