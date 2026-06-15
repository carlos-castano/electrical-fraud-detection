import json
import os
import glob
from pathlib import Path
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import time
from lightgbm import LGBMClassifier, early_stopping as lgbm_early_stopping, log_evaluation
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (make_scorer, matthews_corrcoef, f1_score, precision_score,
                             average_precision_score, recall_score, roc_auc_score, classification_report)
from utils import stratified_train_val_test_split, find_best_threshold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from config import SEED


def evaluate_datasets_baseline(data_dir="data/all_features/"):
    """
    Eval all datasets using base models (LGBM, XGB, CatBoost).
    Uses threshold-independent metrics (PR-AUC and ROC-AUC) via Stratified 5-Fold CV
    exclusively on the training set.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    imbalance_ratio = 10.5
    
    # Modelos base estables (sin over/underfitting)
    models = {
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_ratio,
            random_state=SEED, n_jobs=-1, verbose=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_ratio,
            random_state=SEED, n_jobs=-1, eval_metric='logloss'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            subsample=0.8, colsample_bylevel=0.8, scale_pos_weight=imbalance_ratio,
            random_seed=SEED, thread_count=-1, verbose=False
        )
    }

    # CV y métricas
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    results = []
    
    # Iteración por dataset
    for file_path in csv_files:
        dataset_name = os.path.basename(file_path).replace(".csv", "")
        print(f"Evaluating dataset: {dataset_name}...")
        # Lectura, split
        df = pd.read_csv(file_path)
        train_df, _, _ = stratified_train_val_test_split(df, target_col='FLAG')
        X_train = train_df.drop(columns=['FLAG'])
        y_train = train_df['FLAG']
        num_features = X_train.shape[1]
        dataset_results = {'Dataset': dataset_name, 'Num_Features': num_features}
        
        # Evaluación
        for model_name, model in models.items():
            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            dataset_results[f'{model_name}_PR_AUC'] = np.mean(cv_results['test_pr_auc'])
            dataset_results[f'{model_name}_ROC_AUC'] = np.mean(cv_results['test_roc_auc'])  
        results.append(dataset_results)
        
    results_df = pd.DataFrame(results)
    
    # Votación ("Preferred by") basada en PR-AUC
    preferences = {dataset: [] for dataset in results_df['Dataset']}
    for model_name in models.keys():
        pr_auc_col = f'{model_name}_PR_AUC'
        best_idx = results_df[pr_auc_col].idxmax()
        best_dataset = results_df.loc[best_idx, 'Dataset']
        preferences[best_dataset].append(model_name)

    results_df['Preferred_by'] = results_df['Dataset'].map(
        lambda d: ", ".join(preferences[d]) if preferences[d] else "-"
    )
    
    # Orden por PR-AUC promedio
    pr_auc_cols = [f'{m}_PR_AUC' for m in models.keys()]
    results_df['Mean_PR_AUC_All'] = results_df[pr_auc_cols].mean(axis=1)
    results_df = results_df.sort_values(by='Mean_PR_AUC_All', ascending=False).reset_index(drop=True)
    
    return results_df

def evaluate_resampling_by_model(df, model_name='lightgbm'):
    """
    Evaluate different resampling strategies using a specific model in
    a 5-fold cross-validation on the train set.
    """
    print(f"Evaluating resampling strategies | {model_name.upper()}")
    # Particiones SIEMPRE con SEED de config.py
    train_df, _, _ = stratified_train_val_test_split(df, target_col='FLAG')
    X_train = train_df.drop(columns=['FLAG'])
    y_train = train_df['FLAG']
    ratio = 10.5

    scoring = {
        'pr_auc': 'average_precision',
        'roc_auc': 'roc_auc',
        'mcc': make_scorer(matthews_corrcoef),
        'f1_fraud': make_scorer(f1_score, pos_label=1),
        'precision_fraud': make_scorer(precision_score, pos_label=1),
        'recall_fraud': make_scorer(recall_score, pos_label=1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    kmeans_est = KMeans(n_init='auto', random_state=SEED)

    # Estrategias de remuestreo
    samplers_dict = {
        'Baseline_No_Resampling': None,
        'Class_Weights_Native': 'class_weights',
        'Oversampling_SMOTE': SMOTE(random_state=SEED),
        'Oversampling_ADASYN': ADASYN(random_state=SEED),
        'Oversampling_Borderline1': BorderlineSMOTE(random_state=SEED, kind='borderline-1'),
        'Oversampling_KMeansSMOTE': KMeansSMOTE(random_state=SEED, kmeans_estimator=kmeans_est, cluster_balance_threshold=0.1),
        'Undersampling_NearMissV3': NearMiss(version=3),
        'Hybrid_SMOTETomek': SMOTETomek(random_state=SEED),
        'Hybrid_SMOTEENN': SMOTEENN(random_state=SEED)
    }
    
    # Modelos base con menor aleatoriedad que BorutaShap (+estables)
    def get_evaluator_model(apply_native_weights=False):
        if model_name.lower() == 'lightgbm':
            return LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                scale_pos_weight=ratio if apply_native_weights else 1,
                n_jobs=-1, random_state=SEED, verbose=-1
            )
        elif model_name.lower() == 'xgboost':
            return XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=20,
                colsample_bytree=0.7, subsample=0.8, tree_method='hist',
                scale_pos_weight=ratio if apply_native_weights else 1,
                n_jobs=-1, random_state=SEED, verbosity=0
            )
        elif model_name.lower() == 'catboost':
            return CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, min_data_in_leaf=20,
                rsm=0.7, bootstrap_type='Bernoulli', subsample=0.8,
                scale_pos_weight=ratio if apply_native_weights else 1,
                random_seed=SEED, verbose=0
            )
        else:
            raise ValueError("Model not recognized. Must be 'lightgbm', 'xgboost' or 'catboost'.")

    results = []

    for name, sampler in samplers_dict.items():
        print(f"    {name}...")
        # Construcción del pipeline
        if sampler == 'class_weights':
            model = get_evaluator_model(apply_native_weights=True)
            steps = [('model', model)]
        elif sampler is None:
            model = get_evaluator_model(apply_native_weights=False)
            steps = [('model', model)]
        else:
            model = get_evaluator_model(apply_native_weights=False)
            steps = [('sampler', sampler), ('model', model)]

        pipeline = Pipeline(steps=steps)

        # Evaluación y resultados
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        
        results.append({
            'Model': model_name.upper(),
            'Strategy': name,
            'PR_AUC_mean': np.mean(cv_results['test_pr_auc']),
            'ROC_AUC_mean': np.mean(cv_results['test_roc_auc']),
            'MCC_mean': np.mean(cv_results['test_mcc']),
            'Recall_Fraud_mean': np.mean(cv_results['test_recall_fraud']),
            'Precision_Fraud_mean': np.mean(cv_results['test_precision_fraud']),
            'F1_Fraud_mean': np.mean(cv_results['test_f1_fraud'])
        })
    # Orden por PR-AUC
    results_df = pd.DataFrame(results).sort_values(by='PR_AUC_mean', ascending=False).reset_index(drop=True)
    return results_df

def optimize_model_optuna(df, model_name, n_trials=100):
    """
    Optuna hyperparameter optimization for LightGBM, XGBoost, or CatBoost.
    Optimizes for PR-AUC using internal CV(3) with Early Stopping & Pruning.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"Optuna Optimization for: {model_name.upper()}")
    print(f"Target Metric: PR-AUC | Trials: {n_trials}")
    
    # Siempre sobre train
    train_df, _, _ = stratified_train_val_test_split(df, target_col='FLAG')
    X_train_full = train_df.drop(columns=['FLAG']).reset_index(drop=True)
    y_train_full = train_df['FLAG'].reset_index(drop=True)

    def objective(trial):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        pr_auc_scores = []
        
        # Parámetros a buscar
        if model_name.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'boosting_type': 'gbdt',
                'n_estimators': 3500,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
                'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
                'path_smooth': trial.suggest_float('path_smooth', 0.0, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
                'max_bin': trial.suggest_categorical('max_bin', [255, 512]),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), 
                'random_state': SEED,
                'n_jobs': -1,
                'verbose': -1
            }
            max_leaves = 2 ** params['max_depth'] - 1
            if params['num_leaves'] > max_leaves:
                params['num_leaves'] = max_leaves

        elif model_name.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'n_estimators': 3500,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'max_bin': trial.suggest_categorical('max_bin', [256, 512]),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': 0
            }

        elif model_name.lower() == 'catboost':
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'PRAUC',
                'iterations': 3000, 
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 0.1, 20.0, log=True),
                'rsm': trial.suggest_float('rsm', 0.4, 1.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 5),
                'border_count': trial.suggest_categorical('border_count', [128, 200, 254]),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'random_seed': SEED,
                'verbose': 0,
            }
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.5, 0.9)
        else:
            raise ValueError("Model must be 'lightgbm', 'xgboost', or 'catboost'")

        # CV Interna con Early Stopping y Pruning
        for step, (train_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full)):
            X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            if model_name.lower() == 'lightgbm':
                model = LGBMClassifier(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[
                        lgbm_early_stopping(stopping_rounds=50, verbose=False),
                        log_evaluation(period=0)
                    ]
                )
            
            elif model_name.lower() == 'xgboost':
                model = XGBClassifier(early_stopping_rounds=50, **params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
                
            elif model_name.lower() == 'catboost':
                model = CatBoostClassifier(early_stopping_rounds=50, **params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )

            # Pred y eval
            probs = model.predict_proba(X_fold_val)[:, 1]
            score = average_precision_score(y_fold_val, probs)
            pr_auc_scores.append(score)
            
            # Si el trial no mejora Pruning
            intermediate_value = np.mean(pr_auc_scores)
            trial.report(intermediate_value, step)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(pr_auc_scores)

    # Sampler y Pruner
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1) # en el segundo fold
    
    study_name = f"{model_name.upper()}_Optimization"
    study = optuna.create_study(
        direction='maximize', 
        study_name=study_name, 
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n== BEST RESULTS FOR {model_name.upper()} ==")
    print(f"Best PR-AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}={value}")
        
    return study

def save_optuna_params(study, model_name, save_dir="data/optuna_parameters"):
    """
    Saves FULL model parameters (Optuna + fixed params) into a JSON file.
    Ready to be used directly in model initialization.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    params = study.best_params.copy()
    model_name = model_name.lower()
    # Añade los valores por defecto
    if model_name == 'lightgbm':
        params.update({
            'objective': 'binary',
            'metric': 'average_precision',
            'boosting_type': 'gbdt',
            'n_estimators': 3500,
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1
        })

    elif model_name == 'xgboost':
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'tree_method': 'hist',
            'n_estimators': 3500,
            'random_state': SEED,
            'early_stopping_rounds': 100,
            'n_jobs': -1,
            'verbosity': 0
        })

    elif model_name == 'catboost':
        params.update({
            'loss_function': 'Logloss',
            'eval_metric': 'PRAUC',
            'iterations': 3500,
            'random_seed': SEED,
            'early_stopping_rounds': 100,
            'verbose': 0
        })

    else:
        raise ValueError("model_name must be 'lightgbm', 'xgboost' or 'catboost'")

    file_path = os.path.join(save_dir, f"{model_name}_params.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)
    print(f"Saved {model_name.upper()} parameters to: {file_path}")


def load_optuna_params(model_name, load_dir="data/optuna_parameters"):
    """
    Loads parameters directly usable in model initialization.
    """
    file_path = os.path.join(load_dir, f"{model_name.lower()}_params.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No params file found for {model_name} at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params


def evaluate_best_model(model, model_name,
                        train_df, val_df, test_df,
                        target_col='FLAG',
                        seeds=[SEED, 123, 456, 789, 1011],
                        main_seed=SEED):
    """
    Evaluate the final models using multiple seeds
    - Train the models, find the optimal threshold (validation set), and evaluate them on test.
    - Print the results for main_seed.
    - Return the model, validation probabilities, test probabilities, and the best threshold
      for main_seed for later use in the notebook. Additionally, return two DataFrames containing
      the aggregated metrics across seeds.
    """
    results = []
    training_times = []
    main_outputs = {}

    # Datos
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    print("=" * 60)
    print(f"Evaluating: {model_name.upper()}")

    for seed in seeds:
        params = model.get_params()
        if 'random_state' in params:
            params['random_state'] = seed
        if 'seed' in params:
            params['seed'] = seed
        if 'random_seed' in params:
            params['random_seed'] = seed
        model_seed = model.__class__(**params)

        # Entreno con Early Stopping
        start_time = time.time()

        if model_name.lower() == 'lightgbm':
            model_seed.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgbm_early_stopping(stopping_rounds=100, verbose=False)]
            )
            best_iter = model_seed.best_iteration_

        elif model_name.lower() == 'xgboost':
            model_seed.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            best_iter = model_seed.best_iteration

        elif model_name.lower() == 'catboost':
            model_seed.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            best_iter = getattr(model_seed, 'best_iteration_', model_seed.get_best_iteration())

        end_time = time.time()
        training_times.append(end_time - start_time)

        # Threshold en val
        val_probs = model_seed.predict_proba(X_val)[:, 1]
        best_threshold, best_precision_val, best_mcc_val = find_best_threshold(y_val, val_probs)

        # Test
        test_probs = model_seed.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_threshold).astype(int)
        roc_auc = roc_auc_score(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)
        test_mcc = matthews_corrcoef(y_test, test_preds)

        # Prints solo para main_seed
        if seed == main_seed:
            print(f"\nResults for seed: {seed}")
            print(f"-> Trees Created (Best Iteration): {best_iter}")
            print(f"-> Best Threshold (Precision>=0.55 & Max MCC): {best_threshold:.4f} | Val. Prec: {best_precision_val:.4f}, MCC: {best_mcc_val:.4f}")
            print("\nTEST METRICS")
            print(f"    ROC-AUC: {roc_auc:.4f}")
            print(f"    PR-AUC:  {pr_auc:.4f}")
            print(f"    MCC:     {test_mcc:.4f}")
            print("-" * 60)

            main_outputs = {
                "model": model_seed,
                "val_probs": val_probs,
                "test_probs": test_probs,
                "threshold": best_threshold
            }

        # Métricas
        run_metrics = {
            'Seed': seed,
            'Threshold': best_threshold,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'MCC': test_mcc,
            'F1_Macro': f1_score(y_test, test_preds, average='macro'),
            'Recall_Fraud': recall_score(y_test, test_preds, pos_label=1),
            'Recall_NoFraud': recall_score(y_test, test_preds, pos_label=0),
            'Precision_Fraud': precision_score(y_test, test_preds, pos_label=1, zero_division=0),
            'Precision_NoFraud': precision_score(y_test, test_preds, pos_label=0, zero_division=0)
        }
        results.append(run_metrics)

    # Resultados agregados
    df_results = pd.DataFrame(results)
    summary = df_results.drop('Seed', axis=1).agg(['mean', 'std']).T
    summary['Mean_Std'] = summary.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)
    summary = summary[['Mean_Std']]

    # Tiempo medio
    avg_time = np.mean(training_times)
    print(f"Average training time: {avg_time:.2f} sec")
    print("=" * 60, "\n")

    return (
        main_outputs["model"],
        main_outputs["val_probs"],
        main_outputs["test_probs"],
        main_outputs["threshold"],
        df_results,
        summary
    )


def evaluate_ensembles(val_probs_dict, test_probs_dict, y_val, y_test):
    """
    Evaluate Soft Voting and Stacking (Logistic Regression) using
    pre-calculated probabilities from the base models.
    """
    print("="*60)
    print(" EVALUATING ENSEMBLES (SOFT VOTING & STACKING) ")
    print("="*60)

    # Predicciones de los modelos base
    X_val_meta = np.column_stack(list(val_probs_dict.values()))
    X_test_meta = np.column_stack(list(test_probs_dict.values()))
    
    print("\n\t** SOFT VOTING (Average) **")

    # Umbral en validación
    val_probs_soft = X_val_meta.mean(axis=1)
    test_probs_soft = X_test_meta.mean(axis=1)
    best_th_soft, best_precision_val_soft, best_mcc_val_soft = find_best_threshold(y_val, val_probs_soft)
    print(f"-> Best Threshold (Val): {best_th_soft:.4f} (Prec: {best_precision_val_soft:.4f}, MCC: {best_mcc_val_soft:.4f})")
    
    # Evalúa en test
    test_preds_soft = (test_probs_soft >= best_th_soft).astype(int)
    print(f"    TEST ROC-AUC: {roc_auc_score(y_test, test_probs_soft):.4f}")
    print(f"    TEST PR-AUC:  {average_precision_score(y_test, test_probs_soft):.4f}")
    print(f"    TEST MCC:     {matthews_corrcoef(y_test, test_preds_soft):.4f}")
    print("\nClassification Report (Soft Voting):")
    print(classification_report(y_test, test_preds_soft, digits=4))
    print("-"*60)

    print("\n\t** STACKING (Logistic Regression) **")
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_val_meta, y_val)
    
    # Pesos o coeficientes de los modelos
    print("Logistic Regression Coefficients:")
    for name, coef in zip(val_probs_dict.keys(), meta_model.coef_[0]):
        print(f"  - {name}: {coef:.4f}")
    
    # Mismo umbral
    val_probs_stack = meta_model.predict_proba(X_val_meta)[:, 1]
    test_probs_stack = meta_model.predict_proba(X_test_meta)[:, 1]
    best_th_stack, best_precision_val_stack, best_mcc_val_stack = find_best_threshold(y_val, val_probs_stack)
    print(f"\n Best Threshold (Val): {best_th_stack:.4f} (Prec: {best_precision_val_stack:.4f}, MCC: {best_mcc_val_stack:.4f})")
    
    # Y test
    test_preds_stack = (test_probs_stack >= best_th_stack).astype(int)
    print(f"    TEST ROC-AUC: {roc_auc_score(y_test, test_probs_stack):.4f}")
    print(f"    TEST PR-AUC:  {average_precision_score(y_test, test_probs_stack):.4f}")
    print(f"    TEST MCC:     {matthews_corrcoef(y_test, test_preds_stack):.4f}")
    print("\nClassification Report (Stacking):")
    print(classification_report(y_test, test_preds_stack, digits=4))