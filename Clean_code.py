import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

def chargement_csv(chemin_fichier1, chemin_fichier2):
    """
    Charge et fusionne deux fichiers CSV en un seul DataFrame.

    """
    try:
        # Lecture des fichiers CSV
        df1 = pd.read_csv(chemin_fichier1)
        df2 = pd.read_csv(chemin_fichier2)

        # Fusion des DataFrames
        df = pd.merge(df1, df2, on='respondent_id')

        return df

    except pd.errors.EmptyDataError:
        print("Erreur : L'un des fichiers CSV est vide.")
    except pd.errors.ParserError:
        print("Erreur : Erreur de parsing dans l'un des fichiers CSV.")
    except FileNotFoundError:
        print("Erreur : L'un des fichiers spécifiés n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")

    return None

def descriptive_stats(df):
    """
    Cette fonction permet de prendre connaissance des statistiques descriptives d'un dataframe passé en entrée
    La fonction n'est pas parfaite, le but est de faire quelque chose d'industriel pouvant décrire un dataframe peut importe le nombre de variable et le type des colonnes étudiées.
    
    """
    results = pd.DataFrame()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Statistiques pour les colonnes numériques
            stats = {
                'Moyenne': df[col].mean(),
                'Médiane': df[col].median(),
                'Ecart-type': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            }
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Statistiques pour les colonnes catégoriques
            top_freq = df[col].value_counts().idxmax()
            top_freq_count = df[col].value_counts().max()
            stats = {
                'Nombre unique': df[col].nunique(),
                'Top catégorie': top_freq,
                'Fréquence Top catégorie': top_freq_count
            }
        else:
            # Pour les autres types de données, vous pouvez ajouter plus de conditions ici
            stats = {}

        results[col] = pd.Series(stats)

    return results

"""
Les fonctions suivantes (-plot_) visent à produire des visuels afin de visualier la constitution et la distribution du dataframe passé en entrée
Ces fonctions sont perfectibles,  néanmoins elles sont efficace, et adaptable peux importe le nombre de varaibles et leurs types

"""
def plot_histograms(df):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogramme de {col}')
        plt.show()

def plot_boxplots(df):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.show()

def plot_bar_charts(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        sns.countplot(y=df[col])
        plt.title(f'Fréquences des catégories dans {col}')
        plt.show()

def plot_pie_charts(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Répartition des catégories dans {col}')
        plt.ylabel('')
        plt.show()

def plot_correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice de Corrélation')
    plt.show()

"""
"Main" des fonctions de statistiques descriptives

"""
def statistiques_descriptive_df(df):
    print(descriptive_stats(df))
    plot_boxplots(df)
    plot_histograms(df)
    plot_bar_charts(df)
    plot_pie_charts(df)
    plot_correlation_matrix(df)

"""

Nettoyage de données

"""
def na_clean(df):
    #variable anonimisées, pas interpretables
    df = df.drop(['employment_industry','employment_occupation','hhs_geo_region'], axis = 1) 
    #beaucoup de vides dans cette variable, on remplace pour éviter de perdre trop d'observation avec la commande suivante
    df['health_insurance'] = df['health_insurance'].fillna(0)
     #on supirme les enregistrements vide
    df_cleaned = df.dropna()
    return df_cleaned

"""

Nettoyage des données quantitatives

"""
def var_quant_clean(df):
    #matrice de corrélation dans le notebook
    #les 2 var sont corrélées, on les retravaille afin de les supprimer ensuite
    df['doctor_recc_h1n1'] = df['doctor_recc_h1n1'].fillna(0)
    df['doctor_recc_seasonal'] = df['doctor_recc_seasonal'].fillna(0)
    df['doctor_recc_combined'] = df['doctor_recc_h1n1'] + df['doctor_recc_seasonal']
    df['doctor_recc_combined'] = df['doctor_recc_combined'].replace(2, 1)
    #variable avec un forte correlation avec d'autre, on les supprimes
    df_cleaned = df.drop(['doctor_recc_h1n1', 'doctor_recc_seasonal', 'opinion_seas_risk', 'behavioral_outside_home'], axis=1)
    return df_cleaned

"""

Nettoyage des données qualitatives

"""
def var_qual_clean(df):
    #pas de correlation particuliere
    return df 

"""

Mise en forme des données pour un premier modèle

"""
def data_format(df):
    label_encoder = LabelEncoder()
    #on categorise les variables, on code les différentes modalités
    df['age_group_encoded'] = label_encoder.fit_transform(df['age_group'])
    df = df.drop('age_group', axis = 1)
    df['education_encoded'] = label_encoder.fit_transform(df['education'])
    df = df.drop('education', axis = 1)
    df['race_encoded'] = label_encoder.fit_transform(df['race'])
    df = df.drop('race', axis = 1)
    df['sex_encoded'] = label_encoder.fit_transform(df['sex'])
    df = df.drop('sex', axis = 1)
    df['income_poverty_encoded'] = label_encoder.fit_transform(df['income_poverty'])
    df = df.drop('income_poverty', axis = 1)
    df['marital_status_encoded'] = label_encoder.fit_transform(df['marital_status'])
    df = df.drop('marital_status', axis = 1)
    df['rent_or_own_encoded'] = label_encoder.fit_transform(df['rent_or_own'])
    df = df.drop('rent_or_own', axis = 1)
    df['employment_status_encoded'] = label_encoder.fit_transform(df['employment_status'])
    df = df.drop('employment_status', axis = 1)
    df['census_msa_encoded'] = label_encoder.fit_transform(df['census_msa'])
    df = df.drop('census_msa', axis = 1)
    return df

"""

Segmentation des données

"""
def split_x_y(df):
    X_train = df.drop(['h1n1_vaccine', 'seasonal_vaccine','respondent_id'], axis=1)
    y_train = df[['h1n1_vaccine', 'seasonal_vaccine']]
    return X_train, y_train

"""

Premier modèle servant a sélectionner les variables les plus intéressantes

"""
def model_Random_Forest(X_train,y_train):
    rf = RandomForestClassifier(random_state=42)  # Utilisez RandomForestRegressor pour la régression
    rf.fit(X_train, y_train)  

    # Extraction des importances des variables
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importances.head(10)

"""

Récupération des variables les plus importantes pour un nouveau dataframe

"""
def data_top_var(top,df):
    important_vars = top.iloc[:, 0].tolist()
    df_top = df[important_vars]
    return df_top
"""

Mise en forme du nouveau dataframe pour le modèle à suivre

"""
def data_format_2(df_rlm, y_train):
    for col in df_rlm:
        df_rlm.loc[:, col] = df_rlm[col].astype('category')
    df_dummies_rlm= pd.get_dummies(df_rlm)
    y_train = y_train.reset_index(drop=True) #reset de l'indexcar join impossible sinon
    df_dummies_rlm['h1n1_vaccine'] =y_train['h1n1_vaccine'] 
    df_dummies_rlm['seasonal_vaccine'] =y_train['seasonal_vaccine']
    df_dummies_rlm = df_dummies_rlm.drop('doctor_recc_combined_0.0', axis = 1)
    df_dummies_rlm.fillna(0, inplace=True) 
    return df_dummies_rlm


"""

Première fonction du modèle finale, en commantaire car copy de cette focntion ci-dessous intégrant mlFlow

"""
"""
def models_Logistic_Regression(df_rlm):
    # Séparer les caractéristiques et les variables cibles
    X = df_rlm.drop(['h1n1_vaccine', 'seasonal_vaccine'], axis=1)
    y_h1n1 = df_rlm['h1n1_vaccine']
    y_seasonal = df_rlm['seasonal_vaccine']

    # Division des données pour chaque cible
    X_train_h1n1, X_test_h1n1, y_train_h1n1, y_test_h1n1 = train_test_split(X, y_h1n1, test_size=0.2, random_state=42)
    X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(X, y_seasonal, test_size=0.2, random_state=42)

    # Création et entraînement des modèles de régression logistique
    model_h1n1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model_seasonal = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    model_h1n1.fit(X_train_h1n1, y_train_h1n1)
    model_seasonal.fit(X_train_seasonal, y_train_seasonal)

    # Prédiction des résultats sur les ensembles de test
    predictions_h1n1 = model_h1n1.predict(X_test_h1n1)
    predictions_seasonal = model_seasonal.predict(X_test_seasonal)

###

    # Calculer les probabilités de prédiction pour la courbe ROC
    probabilities_h1n1 = model_h1n1.predict_proba(X_test_h1n1)[:, 1]
    probabilities_seasonal = model_seasonal.predict_proba(X_test_seasonal)[:, 1]

    # Calculer les scores AUC
    auc_h1n1 = roc_auc_score(y_test_h1n1, probabilities_h1n1)
    auc_seasonal = roc_auc_score(y_test_seasonal, probabilities_seasonal)

    # Afficher la courbe ROC
    fpr_h1n1, tpr_h1n1, _ = roc_curve(y_test_h1n1, probabilities_h1n1)
    fpr_seasonal, tpr_seasonal, _ = roc_curve(y_test_seasonal, probabilities_seasonal)

    plt.figure(figsize=(12, 6))
    plt.plot(fpr_h1n1, tpr_h1n1, label=f'h1n1_vaccine AUC: {auc_h1n1:.2f}')
    plt.plot(fpr_seasonal, tpr_seasonal, label=f'seasonal_vaccine AUC: {auc_seasonal:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Afficher la matrice de confusion
    cm_h1n1 = confusion_matrix(y_test_h1n1, predictions_h1n1)
    cm_seasonal = confusion_matrix(y_test_seasonal, predictions_seasonal)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_h1n1, annot=True, fmt='g', ax=ax[0])
    ax[0].set_title('Confusion Matrix for h1n1_vaccine')
    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')

    sns.heatmap(cm_seasonal, annot=True, fmt='g', ax=ax[1])
    ax[1].set_title('Confusion Matrix for seasonal_vaccine')
    ax[1].set_xlabel('Predicted Labels')
    ax[1].set_ylabel('True Labels')
    plt.show() 
###
    # Évaluation des modèles
    print("Évaluation pour h1n1_vaccine:")
    print(classification_report(y_test_h1n1, predictions_h1n1))

    print("Évaluation pour seasonal_vaccine:")
    print(classification_report(y_test_seasonal, predictions_seasonal))
    return 
"""

"""

Régression logistique sur les variables à prédire soit 'h1n1_vaccine' et 'seasonal_vaccine'
et mesure de performance des modèles
Implémentation de MlFlow

"""
# models_Logistic_Regression avec mlflow
def models_Logistic_Regression(df_rlm):
    with mlflow.start_run(run_name="Logistic Regression Models"):
        # Séparer les caractéristiques et les variables cibles
        X = df_rlm.drop(['h1n1_vaccine', 'seasonal_vaccine'], axis=1)
        y_h1n1 = df_rlm['h1n1_vaccine']
        y_seasonal = df_rlm['seasonal_vaccine']

        # Division des données pour chaque cible
        X_train_h1n1, X_test_h1n1, y_train_h1n1, y_test_h1n1 = train_test_split(X, y_h1n1, test_size=0.2, random_state=42)
        X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(X, y_seasonal, test_size=0.2, random_state=42)

        # Création et entraînement des modèles de régression logistique
        model_h1n1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model_seasonal = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

        model_h1n1.fit(X_train_h1n1, y_train_h1n1)
        model_seasonal.fit(X_train_seasonal, y_train_seasonal)

        # Enregistrement des modèles
        mlflow.sklearn.log_model(model_h1n1, "model_h1n1")
        mlflow.sklearn.log_model(model_seasonal, "model_seasonal")

        # Prédiction et évaluation des résultats sur les ensembles de test
        predictions_h1n1 = model_h1n1.predict(X_test_h1n1)
        predictions_seasonal = model_seasonal.predict(X_test_seasonal)
        acc_h1n1 = accuracy_score(y_test_h1n1, predictions_h1n1)
        acc_seasonal = accuracy_score(y_test_seasonal, predictions_seasonal)
        auc_h1n1 = roc_auc_score(y_test_h1n1, model_h1n1.predict_proba(X_test_h1n1)[:, 1])
        auc_seasonal = roc_auc_score(y_test_seasonal, model_seasonal.predict_proba(X_test_seasonal)[:, 1])

        # Enregistrement des métriques
        mlflow.log_metric("accuracy_h1n1", acc_h1n1)
        mlflow.log_metric("accuracy_seasonal", acc_seasonal)
        mlflow.log_metric("auc_h1n1", auc_h1n1)
        mlflow.log_metric("auc_seasonal", auc_seasonal)

        # Enregistrement des rapports de classification
        report_h1n1 = classification_report(y_test_h1n1, predictions_h1n1)
        report_seasonal = classification_report(y_test_seasonal, predictions_seasonal)
        mlflow.log_text(report_h1n1, "classification_report_h1n1.txt")
        mlflow.log_text(report_seasonal, "classification_report_seasonal.txt")

        print("Évaluation pour h1n1_vaccine:")
        print(report_h1n1)
        print("Évaluation pour seasonal_vaccine:")
        print(report_seasonal)
    

def main():
    #chargement donnees
    df = chargement_csv('training_set_features.csv','training_set_labels.csv')
    #descripition  des donnees
    #statistiques_descriptive_df(df)
    #Nettoyage des vides
    df = na_clean(df)
    #nettoyage des données numériques
    #en fonction de la correlatiion, etudié sur le notebook
    df = var_quant_clean(df)
    #nettoyage des données qualitatives
    #en fonction des v de cramers calculés sur le notebook
    df = var_qual_clean(df)
    #statistiques_descriptive_df(df)
    #Mise en forme pour le modèle qui va suivre
    df = data_format(df)
    X_train, y_train = split_x_y(df)
    #Selection des 10 variables les plus partinentes via un Random Forest
    top = model_Random_Forest(X_train, y_train)
    #dataframe avec les 10 variables les plus importantes
    df_rlm = data_top_var(top,df)
    #mise en forme des données pour le modèle
    df_rlm = data_format_2(df_rlm, y_train)
    #modèle final
    models_Logistic_Regression(df_rlm)
    return

main()