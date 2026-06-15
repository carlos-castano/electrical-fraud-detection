import os
import json
import pandas as pd
from utils import stratified_train_val_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
###########################################################################################
# No cambiar el orden de los imports
# Ver hilo de incompatibilidad en: https://github.com/Ekeany/Boruta-Shap/issues/130
import scipy.stats as stats
if not hasattr(stats, "binom_test"):
    stats.binom_test = lambda x, n, p, alternative: stats.binomtest(k=int(x), n=int(n), p=p, alternative=alternative).pvalue

np.NaN = np.nan # Necesario con NumPy >= 2.0
from BorutaShap import BorutaShap
###########################################################################################
from config import SEED

def apply_boruta_filtering(df, boruta_json_path):
    """
    Filter a DataFrame using the features previously selected with BorutaShap.
    """
    if not os.path.exists(boruta_json_path):
        raise FileNotFoundError(
            f"The BorutaShap selection could not be found for this dataset:\n{boruta_json_path}"
        )
    # Carga y filtrado
    with open(boruta_json_path, "r") as f:
        selected_features = json.load(f)
    df_filtered = df[selected_features + ['FLAG']]

    return df_filtered

def compute_boruta_selection_by_model(path_csv, model_name='lightgbm',
                                      output_path='data/all_features/boruta_feature_selection', n_trials=100):
    """
    Run BorutaShap on the training dataset using a specific model set as 'Explorator'
    and save the selected features in JSON.
    """
    print(f"Running BorutaShap | {model_name.upper()} | {os.path.basename(path_csv)}")
    # Carga y split
    df = pd.read_csv(path_csv)
    train_df, _, _ = stratified_train_val_test_split(df)
    X_train = train_df.drop(columns='FLAG')
    y_train = train_df['FLAG']
    imbalance_ratio = 10.5

    # Configuración "exploradora" (mayor aleatoriedad y bajo sobreajuste)
    if model_name.lower() == 'lightgbm':
        model_boruta = LGBMClassifier(
            n_estimators=200, max_depth=5, num_leaves=20, min_child_samples=30,
            colsample_bytree=0.5, feature_fraction_bynode=0.8, subsample=0.8, subsample_freq=1,
            extra_trees=True, scale_pos_weight=imbalance_ratio, n_jobs=-1, random_state=SEED, verbose=-1
        )
    elif model_name.lower() == 'xgboost':
        model_boruta = XGBClassifier(
            n_estimators=200, max_depth=5, min_child_weight=20, colsample_bynode=0.8,
            colsample_bytree=0.5, subsample=0.8, scale_pos_weight=imbalance_ratio,
            tree_method='hist', n_jobs=-1, random_state=SEED, verbosity=0
        )
    # ver Catboost: https://catboost.ai/docs/en/references/training-parameters/common
    elif model_name.lower() == 'catboost':
        model_boruta = CatBoostClassifier(
            iterations=200, depth=5, min_data_in_leaf=30,
            # rsm: "percentage of features to use at each split selection"
            rsm=0.5, bootstrap_type='Bernoulli', subsample=0.8,
            scale_pos_weight=imbalance_ratio, random_seed=SEED, verbose=0
        )
    else:
        raise ValueError("Model must be 'lightgbm', 'xgboost', or 'catboost'")

    # BorutaShap
    boruta_selector = BorutaShap(
        model=model_boruta,
        importance_measure='shap',
        classification=True,
        pvalue=0.1, # Se aumenta de 0.05 para ser algo más permisivo
    )
    
    boruta_selector.fit(
        X=X_train, y=y_train, n_trials=n_trials, random_state=SEED,
        sample=False, train_or_test='train', normalize=True, verbose=False
    )
    
    # Se aceptan las variables tentativas
    selected_features = list(dict.fromkeys(list(boruta_selector.accepted) + list(boruta_selector.tentative)))
    print(f'Accepted features for {model_name.upper()}: {len(selected_features)}')

    # Guardado
    os.makedirs(output_path, exist_ok=True)
    dataset_name = os.path.basename(path_csv).replace(".csv", "")
    json_name = f"{model_name.lower()}_boruta_{dataset_name}.json"
    json_path = os.path.join(output_path, json_name)
    with open(json_path, "w") as f:
        json.dump(selected_features, f, indent=4)
    print(f"Successfully saved to: {json_path}\n")