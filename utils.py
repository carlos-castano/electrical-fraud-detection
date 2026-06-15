import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_score
from sklearn.model_selection import train_test_split
from config import SEED

def drop_industrial_client(df):
    """Remove the known industrial outlier client."""
    return df[df['CONS_NO'] != 'AC56D4A8437DDAC84F2F7834D7663842'].reset_index(drop=True)

def drop_clientID(df):
    """Drop the 'CONS_NO' (Client ID) column if it exists."""
    if 'CONS_NO' in df.columns:
        return df.drop(columns=['CONS_NO'])
    return df

def drop_duplicated_rows(df):
    """Drop duplicated rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)

def drop_inactive_clients(df):
    """Drop rows with only 0 values or nulls (ignores FLAG column)."""
    is_inactive = (df.iloc[:, :-1].fillna(0) == 0).all(axis=1)
    return df[~is_inactive].reset_index(drop=True)


def drop_n_not_consecutive_values(df, n, target='FLAG'):
    """Drop clients that do not have at least n consecutive non-null values."""
    # Sin considerar la variable objetivo se detectan los nulos
    features = df.drop(columns=[target])
    notnull = features.notnull().astype(int)
    # Se calcula la racha máxima por fila
    def max_consecutive(row):
        groups = (row != row.shift()).cumsum()
        streaks = row.groupby(groups).cumsum()
        return streaks.max()
    # Y las filtra
    max_streaks = notnull.apply(max_consecutive, axis=1)
    mask = max_streaks >= n
    return df[mask].reset_index(drop=True)

def interpolate_spikes(df, extreme_q=0.99, base_q=0.95, multiplier=3, min_absolute=100):
    """
    Substitute extreme spikes with NaN and interpolate linearly (max 2 days).
    """
    flag_col = df['FLAG']
    df_dates = df.drop(columns=['FLAG']).copy()
    # Filtro de valores extremos (i)
    q_extreme = df_dates.quantile(extreme_q, axis=1)
    # Umbral percentil base * factor (filtra -> debe ser muy extremo para el cliente (ii))
    q_base = df_dates.quantile(base_q, axis=1)
    scale_threshold = q_base * multiplier
    # Filtro para las 3 condiciones (i, ii, iii), siendo iii un mínimo de consumo
    final_threshold = np.maximum(q_extreme, scale_threshold)
    final_threshold = np.maximum(final_threshold, min_absolute)
    # Aplica máscara e interpola
    spikes_mask = df_dates.gt(final_threshold, axis=0)
    df_dates = df_dates.mask(spikes_mask, np.nan)
    df_dates = df_dates.interpolate(method='linear', axis=1, limit_area='inside', limit=2)
    # Reconstrucción del dataset
    df_dates['FLAG'] = flag_col
    return df_dates

def drop_large_max_gaps(df, max_gap_threshold):
    """
    Drop rows whose maximum internal gap (consecutive NaNs between first and last non-null)
    is >= max_gap_threshold.
    """
    is_null = df.isna()
    # Primer y último no nulo por cliente
    first_valid = (~is_null).idxmax(axis=1)
    last_valid = (~is_null.iloc[:, ::-1]).idxmax(axis=1)
    # sus posiciones
    col_pos = {col: i for i, col in enumerate(df.columns)}
    first_idx = first_valid.map(col_pos)
    last_idx = last_valid.map(col_pos)
    # Y se filtra la zona interna
    cols = np.arange(df.shape[1])
    internal_mask = (cols >= first_idx.values[:, None]) & (cols <= last_idx.values[:, None])
    internal_nulls = is_null & internal_mask
    # Para detectar rachas internas y su máximo para filtrar
    def max_consecutive_nans(row):
        groups = (row != row.shift()).cumsum()
        streaks = row.groupby(groups).cumsum()
        return streaks.max()

    max_gaps = internal_nulls.apply(max_consecutive_nans, axis=1)
    return df[max_gaps < max_gap_threshold].reset_index(drop=True)

def get_trimmed_series(df):
    """Convert the time series into a list of trimmed arrays, removing leading and trailing NaNs."""
    series = df.drop(columns=['FLAG']).values        
    trimmed_list = []

    for row in series:
        not_nan = ~np.isnan(row)
        first = np.argmax(not_nan)
        last = len(row) - 1 - np.argmax(not_nan[::-1])
        trimmed_list.append(row[first:last+1])

    return trimmed_list

def get_imputed_trimmed_series(df, max_inner_gap=7, min_len=180):
    """
    Impute internal NaNs and return a list of trimmed values (1D numpy arrays).
    
    Logic:
    1. Identify the active period for each client.
    2. Linearly interpolate internal gaps <= max_inner_gap.
    3. Impute remaining internal gaps using the client's monthly median.
    4. Use the customer's global median if the monthly median is not available.
    5. Pad series shorter than min_len symmetrically with edge values.
    Return only the active periods as a list of arrays.
    """
    ts_df = df.drop(columns=['FLAG'])
    dates = pd.to_datetime(ts_df.columns, format='%m/%d/%Y')
    months = dates.month
    
    # 1.Límites de actividad
    has_valid_before = ts_df.notna().cumsum(axis=1) > 0
    has_valid_after = ts_df.iloc[:, ::-1].notna().cumsum(axis=1).iloc[:, ::-1] > 0
    is_internal = has_valid_before & has_valid_after
    
    # 2. Interpolación lineal para gaps cortos
    ts_imputed = ts_df.interpolate(method='linear', axis=1, limit=max_inner_gap, limit_area='inside')
    
    # Mediana global del cliente
    client_global_medians = ts_df.median(axis=1)
    
    # 3, 4. Imputación por mediana mensual (en su defecto global) para gaps largos
    for m in range(1, 13):
        month_cols = ts_df.columns[months == m]
        if len(month_cols) == 0:
            continue
            
        client_monthly_median = ts_df[month_cols].median(axis=1)
        client_monthly_median = client_monthly_median.fillna(client_global_medians)
        ts_imputed[month_cols] = ts_imputed[month_cols].apply(lambda col: col.fillna(client_monthly_median))
    
    # Arrays recortadas
    trimmed_imputed_list = []
    is_internal_arr = is_internal.values
    final_vals = ts_imputed.values 
    
    # 5. Padding bidireccional
    for i in range(len(final_vals)):
        active_series = final_vals[i][is_internal_arr[i]]
        n = len(active_series)
        if n < min_len:
            missing_days = min_len - n
            pad_front = missing_days // 2
            pad_back = missing_days - pad_front
            active_series = np.pad(active_series, (pad_front, pad_back), mode='edge')
        trimmed_imputed_list.append(active_series)
        
    return trimmed_imputed_list

def dedup_columns(cols):
    """It makes duplicate columns unique by adding a numerical suffix."""
    seen = {}
    result = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result

def stratified_train_val_test_split(df, target_col='FLAG', train_size=0.70,
                                    val_size=0.15, test_size=0.15, random_state=SEED):
    """
    Split a dataframe into train, validation and test. Stratified by the target_col.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size NOT 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[target_col],
        random_state=random_state
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[target_col],
        random_state=random_state
    )

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def find_best_threshold(y_true, probs, min_precision=0.55):
    """
    Find the optimal threshold that maximizes MCC while ensuring precision >= min_precision.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions = []
    mccs = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precisions.append(precision_score(y_true, preds, pos_label=1, zero_division=0))
        mccs.append(matthews_corrcoef(y_true, preds))

    precisions = np.array(precisions)
    mccs = np.array(mccs)
    valid_idx = np.where(precisions >= min_precision)[0]

    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(mccs[valid_idx])]
    else:
        best_idx = np.argmax(mccs)

    return thresholds[best_idx], precisions[best_idx], mccs[best_idx]