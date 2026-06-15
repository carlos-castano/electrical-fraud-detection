import pandas as pd
import numpy as np
import utils
import os
from scipy.stats import kurtosis, skew, entropy
from statsmodels.tsa.stattools import acf
import tsfel
from tsfel.feature_extraction.calc_features import calc_window_features
from tsfel.utils.signal_processing import correlated_features
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from config import SEED, MIN_CONSECUTIVE_DAYS

def extract_entry_and_lifespan(df):
    """
    Calculate relative entry point and lifespan (in days) for each client.
    Returns a DataFrame.
    """
    # Columnas
    ts_columns = df.columns[:-1]
    total_days = len(ts_columns)
    # Primer y último no nulo
    notna_matrix = df[ts_columns].notna().values
    first_idx = notna_matrix.argmax(axis=1)
    last_idx = total_days - 1 - np.flip(notna_matrix, axis=1).argmax(axis=1)
    # Relative entry point (normalizado)
    relative_entry = first_idx / (total_days - 1)
    # Lifespan en días
    dates = pd.to_datetime(ts_columns, format='%m/%d/%Y')
    lifespan_days = (dates[last_idx] - dates[first_idx]).days
    return pd.DataFrame({
        'relative_entry_point': relative_entry,
        'lifespan': lifespan_days
    }, index=df.index)


def _get_streaks(mask):
    """
    Calculate lengths, start indices, and end indices of True streaks in a boolean array.
    """
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    # Devuelve: longitudes de racha, índices de inicio, índices de fin
    return ends - starts, starts, ends

def extract_zero_features(X_trimmed):
    """
    Extract advanced features related to zero values from trimmed arrays.
    Focuses on ratios and normalized metrics.
    """
    features = []
    for arr in X_trimmed:
        n = len(arr)
        is_zero = (arr == 0)
        zero_idx = np.where(is_zero)[0]
        n_zero = len(zero_idx)

        # Si no hay ceros, rellenar con 0
        if n_zero == 0:
            features.append({k: 0.0 for k in [
                'zeros_ratio', 'zeros_max_streak', 'zeros_mean_streak', 'zeros_std_streak', 
                'zeros_median_streak', 'zeros_spread_ratio', 'zeros_first_half_ratio', 
                'zeros_second_half_ratio', 'zeros_center_of_mass', 'zero_trans_ratio', 
                'zero_stay_prob', 'long_zero_streak_ratio', 'zero_blocks_ratio', 
                'zero_gap_mean', 'zero_gap_std', 'zero_gap_max', 'recent_30d_zero_ratio',
                'recent_60d_zero_ratio', 'zero_longest_streak_ratio'
            ]})
            continue

        streaks, starts, ends = _get_streaks(is_zero)
        
        # Distribución temporal y espacial
        half = n // 2
        first_half = is_zero[:half].sum() / half
        second_half = is_zero[half:].sum() / (n - half)
        spread = (zero_idx[-1] - zero_idx[0] + 1) / n
        center_of_mass = zero_idx.mean() / n
        
        # Densidad en el tramo final de la serie
        recent_30d = is_zero[-30:].sum() / 30.0
        recent_60d = is_zero[-60:].sum() / 60.0
        
        # Transiciones (volatilidad del estado cero)
        diff = np.diff(is_zero.astype(int))
        trans_ratio = np.sum(diff != 0) / n
        stay_prob = np.sum((is_zero[:-1] == True) & (is_zero[1:] == True)) / n_zero

        # Gaps (distancia entre ceros)
        if n_zero > 1:
            gaps = np.diff(zero_idx)
            gap_mean, gap_std, gap_max = gaps.mean(), gaps.std(), gaps.max()
        else:
            gap_mean = gap_std = gap_max = 0

        # Guardado de variables
        features.append({
            'zeros_ratio': n_zero / n,
            'zeros_max_streak': streaks.max(),
            'zeros_mean_streak': streaks.mean(),
            'zeros_std_streak': streaks.std(),
            'zeros_median_streak': np.median(streaks),
            'zeros_spread_ratio': spread,
            'zeros_first_half_ratio': first_half,
            'zeros_second_half_ratio': second_half,
            'zeros_center_of_mass': center_of_mass,
            'zero_trans_ratio': trans_ratio,
            'zero_stay_prob': stay_prob,
            'long_zero_streak_ratio': np.sum(streaks >= 7) / len(streaks),
            'zero_blocks_ratio': len(streaks) / n,
            'zero_gap_mean': gap_mean,
            'zero_gap_std': gap_std,
            'zero_gap_max': gap_max,
            'recent_30d_zero_ratio': recent_30d,
            'recent_60d_zero_ratio': recent_60d,
            'zero_longest_streak_ratio': streaks.max() / n
        })

    return pd.DataFrame(features)

def extract_null_features(X_trimmed):
    """
    Compute summary features that capture the presence, distribution and
    temporal dynamics of NaN values in trimmed time series.
    """
    features = []
    for arr in X_trimmed:
        # Máscara y conteos
        n = len(arr)
        m = np.isnan(arr)
        count_nulls = m.sum()

        if count_nulls == 0:
            features.append({
                'nulls_ratio': 0,'nulls_streaks_count': 0,
                'nulls_max_consecutive': 0,'nulls_mean_consecutive': 0,
                'nulls_short_gaps_ratio': 0, 'nulls_long_gaps_ratio': 0,
                'nulls_abs_step_change_mean': 0,'nulls_step_change_mean': 0,
                'nulls_step_drop_ratio': 0,'nulls_step_rise_ratio': 0,
                'nulls_fragmentation': 0,
                'nulls_inter_gap_dist_mean': 0, 'nulls_entropy': 0,
                'nulls_first_idx_ratio': 0, 'nulls_last_idx_ratio': 0,
                'nulls_temporal_skew': 0,'nulls_longest_gap_ratio': 0,
                'nan_start_prob':0,
            })
            continue

        streaks, starts, ends = _get_streaks(m)
        null_indices = np.where(m)[0]

        # Ratios de gaps cortos y largos
        short_gaps_ratio = np.sum(streaks <= 7) / len(streaks)
        long_gaps_ratio = np.sum(streaks > 30) / len(streaks)

        # Distribución: inicio, fin, asimetría (dirección sobre el centro)
        first_idx_ratio = null_indices[0]/n
        last_idx_ratio = null_indices[-1]/n
        temporal_skew = (null_indices.mean()-(n/2))/(n/2)

        # Fragmentación
        fragmentation = len(streaks)/count_nulls
        longest_gap_ratio = streaks.max()/n

        # Promedio de las distancias entre gaps 
        if len(streaks)>1:
            inter_gaps = starts[1:] - ends[:-1]
            inter_gap_dist = inter_gaps.mean()/n
        else:
            inter_gap_dist = 0

        # Entropía: ¿están dispersos?
        bins = 5
        hist,_ = np.histogram(null_indices,bins=bins,range=(0,n))
        p = hist/hist.sum()
        p = p[p>0] # Se evita log(0)=-inf
        entropy = -(p*np.log(p)).sum()

        # Se guardan los valores previos y posteriores a los gaps
        valid_before = starts>0
        valid_after = ends<n
        vals_before = arr[starts[valid_before]-1]
        vals_after = arr[ends[valid_after]]
        # Sobre estos
        if len(vals_before)>0 and len(vals_after)>0:
            min_len = min(len(vals_before),len(vals_after))
            vals_before = vals_before[:min_len]
            vals_after = vals_after[:min_len]
            # Cambios entre rachas de nulos
            step_changes = vals_after - vals_before
            # Magnitud media del cambio, su dirección y proporción de caídas
            abs_step_change = np.mean(np.abs(step_changes))
            step_change_mean = np.mean(step_changes)
            step_drop_ratio = np.mean(step_changes < 0)
            step_rise_ratio = np.mean(step_changes > 0)
        else:
            abs_step_change=0
            step_change_mean=0
            step_drop_ratio=0
            step_rise_ratio=0

        # Probabilidad de pasar de no-NaN a NaN (inicio de gap)
        diff = np.diff(m.astype(int)) # 1 (de no-NaN a NaN), -1 (de NaN a no-NaN), 0 (sin cambio)
        nan_starts = np.sum(diff==1)
        nan_start_prob = nan_starts/(n-count_nulls) if (n-count_nulls)>0 else 0

        # Return
        features.append({
            'nulls_ratio':count_nulls/n, 'nulls_streaks_count':len(streaks),
            'nulls_max_consecutive':streaks.max(),
            'nulls_mean_consecutive':streaks.mean(),
            'nulls_short_gaps_ratio':short_gaps_ratio,
            'nulls_long_gaps_ratio':long_gaps_ratio,
            'nulls_first_idx_ratio':first_idx_ratio,
            'nulls_last_idx_ratio':last_idx_ratio,
            'nulls_temporal_skew':temporal_skew,
            'nulls_longest_gap_ratio':longest_gap_ratio,
            'nulls_entropy':entropy, 'nulls_abs_step_change_mean':abs_step_change,
            'nulls_step_change_mean':step_change_mean,
            'nulls_step_drop_ratio':step_drop_ratio,
            'nulls_step_rise_ratio':step_rise_ratio,
            'nulls_fragmentation':fragmentation, 'nulls_inter_gap_dist_mean':inter_gap_dist,
            'nan_start_prob':nan_start_prob,
        })

    return pd.DataFrame(features)


def extract_seasonal_features(df):
    """
    Extract seasonal features focusing on Summer vs Winter consumption shifts, 
    YoY growth, and intra-season volatility.
    """
    ts_df = df.drop(columns=['FLAG']) if 'FLAG' in df.columns else df
    dates = pd.to_datetime(ts_df.columns, format='%m/%d/%Y')
    ts_T = ts_df.T
    ts_T.index = dates
    # Segmentación estacional
    summer_mask = ts_T.index.month.isin([5, 6, 7, 8, 9, 10])
    winter_mask = ts_T.index.month.isin([11, 12, 1, 2, 3, 4])
    summer_data = ts_T[summer_mask]
    winter_data = ts_T[winter_mask]
    features = pd.DataFrame(index=ts_df.index)

    # Dispersión, asimetría y Kurtosis estacional (fraude con picos)
    features['summer_std'] = summer_data.std().fillna(0).values
    features['winter_std'] = winter_data.std().fillna(0).values
    features['summer_skew'] = summer_data.skew().fillna(0).values
    features['winter_skew'] = winter_data.skew().fillna(0).values
    features['summer_kurtosis'] = summer_data.kurtosis().fillna(0).values
    features['winter_kurtosis'] = winter_data.kurtosis().fillna(0).values
    # MASD (media absoluta de la diferencia sucesiva: ¿variable o estable?)
    features['summer_masd'] = summer_data.diff().abs().mean().fillna(0).values
    features['winter_masd'] = winter_data.diff().abs().mean().fillna(0).values
    # Medias y ratios
    summer_mean = summer_data.mean().fillna(0)
    winter_mean = winter_data.mean().fillna(0)
    features['sum_win_mean_ratio'] = (summer_mean / (winter_mean + 1e-5)).values
    features['sum_win_max_ratio'] = (summer_data.max().fillna(0) / (winter_data.max().fillna(0) + 1e-5)).values
    total_sum = ts_T.sum().fillna(0)
    features['summer_energy_prop'] = (summer_data.sum().fillna(0) / (total_sum + 1e-5)).values
    # Picos
    features['summer_p95'] = summer_data.quantile(0.95).fillna(0).values
    features['winter_p05'] = winter_data.quantile(0.05).fillna(0).values
    features['seasonal_gap_robust'] = features['summer_p95'] - features['winter_p05']
    # Análisis YoY
    s14 = summer_data[summer_data.index.year == 2014].mean().fillna(0)
    s15 = summer_data[summer_data.index.year == 2015].mean().fillna(0)
    s16 = summer_data[summer_data.index.year == 2016].mean().fillna(0)
    w14 = winter_data[winter_data.index.year == 2014].mean().fillna(0)
    w16 = winter_data[winter_data.index.year == 2016].mean().fillna(0)
    features['summer_yoy_15_14'] = (s15 / (s14 + 1e-5)).values
    features['summer_yoy_16_15'] = (s16 / (s15 + 1e-5)).values
    features['summer_slope'] = (s16 - s14).values / 2.0
    features['winter_slope'] = (w16 - w14).values / 2.0
    # Divergencia de crecimiento y tendencias
    features['sum_win_slope_diff'] = features['summer_slope'] - features['winter_slope']
    m14 = ts_T[ts_T.index.year == 2014].mean().fillna(0)
    m15 = ts_T[ts_T.index.year == 2015].mean().fillna(0)
    m16 = ts_T[ts_T.index.year == 2016].mean().fillna(0)
    features['annual_slope'] = (m16 - m14).values / 2.0
    features['trend_acceleration'] = ((m16 - m15) - (m15 - m14)).values
    monthly_means = ts_T.groupby(ts_T.index.month).mean()
    features['max_monthly_shift'] = monthly_means.diff().abs().max().fillna(0).values
    features['annual_cv'] = (ts_T.std() / (ts_T.mean() + 1e-5)).fillna(0).values
    features['peak_month'] = monthly_means.idxmax().fillna(0).values
    features['lowest_month'] = monthly_means.idxmin().fillna(0).values
    # Consumo meses pico vs resto
    normal_mask = ts_T.index.month.isin([1, 2, 3, 4, 5, 10, 11, 12])
    heavy_mask = ts_T.index.month.isin([6, 7, 8, 9])
    normal_mean = ts_T[normal_mask].mean().fillna(0)
    heavy_mean = ts_T[heavy_mask].mean().fillna(0)
    features['transitional_ratio'] = (heavy_mean / (normal_mean + 1e-5)).values

    return features.reset_index(drop=True)


def extract_consumption_features(X_trimmed, n_windows):
    """
    Extract advanced consumption features: volatility, shape, signal processing, 
    and other highly detailed window-based extraction.
    """
    if n_windows < 1:
        n_windows = 1
    elif n_windows > 12:
        n_windows = 12

    # Función auxiliar para rachas
    def get_max_streak(mask):
        if not np.any(mask): return 0
        padded = np.pad(mask, (1, 1), 'constant')
        starts = np.where(np.diff(padded) == 1)[0]
        ends = np.where(np.diff(padded) == -1)[0]
        return np.max(ends - starts)

    features = []
    
    for arr in X_trimmed:
        vals = arr[~np.isnan(arr)]
        n = len(vals)           
        # 1. Métricas Globales
        mean_v = np.mean(vals)
        median_v = np.median(vals)
        std_v = np.std(vals)
        q25, q75, p90, p95, p99 = np.percentile(vals, [25, 75, 90, 95, 99])
        diffs = np.diff(vals)
        masd = np.mean(np.abs(diffs))
        # Energía y crossing rate
        energy = np.sum(vals**2)
        mean_crossings = np.sum(np.diff(vals > mean_v) != 0) / n
        # Transformada de Fourier
        fft_vals = np.abs(np.fft.fft(vals - mean_v))[:n//2]
        if np.sum(fft_vals) > 0:
            p_fft = fft_vals / np.sum(fft_vals)
            spectral_entropy = -np.sum(p_fft[p_fft > 0] * np.log(p_fft[p_fft > 0]))
            dominant_freq = np.max(fft_vals)
        else:
            spectral_entropy = dominant_freq = 0.0
        # Varianza de la media móvil
        rolling_mean_30 = np.convolve(vals, np.ones(30)/30, mode='valid')
        rolling_var_30d = np.var(rolling_mean_30)
        # Entropía de Shannon
        hist_counts, _ = np.histogram(vals, bins='auto', density=True)
        shannon_ent = entropy(hist_counts + 1e-9)
        # Autocorrelación con retardos 1, 7 y 30
        if std_v > 1e-5:
            try:
                acf_vals = acf(vals, nlags=30, fft=True)
                acf_1 = acf_vals[1] if len(acf_vals) > 1 and not np.isnan(acf_vals[1]) else 0.0
                acf_7 = acf_vals[7] if len(acf_vals) > 7 and not np.isnan(acf_vals[7]) else 0.0
                acf_30 = acf_vals[30] if len(acf_vals) > 30 and not np.isnan(acf_vals[30]) else 0.0
            except:
                acf_1 = acf_7 = acf_30 = 0.0
        else:
            acf_1 = acf_7 = acf_30 = 0.0
        # Rachas monótonas de crecimiento o decrecimiento
        max_mono_inc = get_max_streak((diffs > 0).astype(int))
        max_mono_dec = get_max_streak((diffs < 0).astype(int))
        # Proporción de días que apenas cambian de la media (>0)
        is_flat = ((np.abs(diffs) < 0.01 * mean_v) & (vals[1:] > 0)).astype(int)
        flat_days_ratio = np.mean(is_flat) if len(is_flat) > 0 else 0
        # ¿Es un consumo extrañamente plano? (variable categórica ordinal)
        max_plateau = get_max_streak(is_flat)
        plateau_type = 0 if max_plateau <= 5 else (1 if max_plateau <= 10 else 2)
        # Caídas del >30% seguidas inmediatamente de subidas del >30% (o viceversa)
        oscillation_thresh = 0.3 * mean_v
        if len(diffs) > 1:
            erratic_rebounds = np.sum((diffs[:-1] < -oscillation_thresh) & (diffs[1:] > oscillation_thresh)) + \
                               np.sum((diffs[:-1] > oscillation_thresh) & (diffs[1:] < -oscillation_thresh))
        else:
            erratic_rebounds = 0
        # Asimetría de las diferencias
        diff_skew = skew(diffs) if np.var(diffs) > 1e-5 else 0.0

        base_feats = {
            'cons_global_mean': mean_v,
            'cons_global_cv': std_v / (mean_v + 1e-5),
            'cons_global_skewness': skew(vals) if std_v > 0 else 0.0,
            'cons_global_kurtosis': kurtosis(vals) if std_v > 0 else 0.0,
            'cons_global_iqr': q75 - q25,
            'cons_global_percentile_spread': p95 / (p90 + 1e-5),
            'cons_global_extreme_peak_ratio': p99 / (median_v + 1e-5),
            'cons_global_masd': masd,
            'cons_global_diff_var': np.var(diffs),
            'cons_global_trend_slope': np.polyfit(np.arange(n), vals, 1)[0],
            'cons_global_sudden_drops_count': np.sum((vals[1:] < 0.5 * vals[:-1]) & (vals[:-1] > 0)),
            'cons_global_sudden_spikes_count': np.sum((vals[1:] > 2.0 * vals[:-1]) & (vals[:-1] > 0)),
            'cons_global_spectral_entropy': spectral_entropy,
            'cons_global_dominant_freq': dominant_freq,
            'cons_global_rolling_var_30d': rolling_var_30d,
            'cons_global_shannon_entropy': shannon_ent,
            'cons_global_energy': energy,
            'cons_global_mean_crossings': mean_crossings,
            'cons_global_acf_1': acf_1,
            'cons_global_acf_7': acf_7,
            'cons_global_acf_30': acf_30,
            'cons_global_max_monotonic_increase': max_mono_inc,
            'cons_global_max_monotonic_decrease': max_mono_dec,
            'cons_global_flat_days_ratio': flat_days_ratio,
            'cons_global_erratic_rebounds': erratic_rebounds,
            'cons_global_diff_skewness': diff_skew,
            'plateau_type': plateau_type
        }
        
        # 2. Métricas por ventana
        if n_windows > 1:
            chunks = np.array_split(vals, n_windows)
            # Listas para capturar variabilidad entre ventanas
            list_win_cvs, list_win_masds, list_win_means = [], [], []

            for i, chunk in enumerate(chunks, 1):
                if len(chunk) < 2: continue
                # Estadísticas básicas de la ventana actual (win_)
                win_mean = np.mean(chunk)
                win_std = np.std(chunk)
                win_median = np.median(chunk)
                win_max = np.max(chunk)
                win_min = np.min(chunk)
                win_p95 = np.percentile(chunk, 95)
                win_diffs = np.diff(chunk)
                win_masd = np.mean(np.abs(win_diffs)) if len(win_diffs) > 0 else 0
                win_cv = win_std / (win_mean + 1e-5)

                # append para el análisis entre ventanas
                list_win_cvs.append(win_cv)
                list_win_masds.append(win_masd)
                list_win_means.append(win_mean)
                
                base_feats.update({
                    f'w{i}_cv': win_cv,
                    f'w{i}_skewness': skew(chunk) if np.isfinite(skew(chunk)) else 0.0,
                    f'w{i}_kurtosis': kurtosis(chunk) if np.isfinite(kurtosis(chunk)) else 0.0,
                    f'w{i}_peak_ratio': win_p95 / (win_median + 1e-5),
                    f'w{i}_masd': win_masd,
                    f'w{i}_trend_slope': np.polyfit(np.arange(len(chunk)), chunk, 1)[0],
                    f'w{i}_energy_ratio': np.sum(chunk**2) / (energy + 1e-5),
                    f'w{i}_spread': win_max - win_min,
                    f'w{i}_mean_crossings': np.sum(np.diff(chunk > win_mean) != 0) / len(chunk),
                })
            
            # Estudio entre ventanas
            if len(list_win_cvs) > 1:
                # ¿Son estables?
                base_feats['windows_cv_volatility'] = np.std(list_win_cvs)
                base_feats['windows_masd_max_ratio'] = np.max(list_win_masds) / (np.min(list_win_masds) + 1e-5)
                base_feats['windows_mean_shift'] = np.max(list_win_means) / (np.min(list_win_means) + 1e-5)

        features.append(base_feats)
        
    return pd.DataFrame(features)

def _raw_dataset_4_tsfel(df_raw, max_gap_threshold=670):
    """Preprocess the raw dataset for TSFEL extraction"""
    df = (df_raw
        .pipe(utils.drop_industrial_client)
        .pipe(utils.drop_clientID)
        .pipe(utils.drop_duplicated_rows)
        .pipe(utils.drop_inactive_clients)
        .pipe(utils.drop_n_not_consecutive_values, n=MIN_CONSECUTIVE_DAYS)
        .pipe(utils.drop_large_max_gaps, max_gap_threshold=max_gap_threshold)
    )
    df_no_errors = utils.interpolate_spikes(df)
    X_trimmed_imputed = utils.get_imputed_trimmed_series(df_no_errors)
    return df_no_errors, X_trimmed_imputed

def create_and_save_tsfel_datasets(df_raw, output_path='data/tsfel_datasets/'):
    """
    Run TSFEL extraction and generate datasets for max_gap_threshold=[670, 250].
    """
    # Los valores max_gap_threshold solo afectan al filtrado posterior de la extracción
    # por lo que se realiza sobre el menos restrictivo para 1 solo cálculo
    print("Preprocessing raw dataset for TSFEL extraction...")
    df_670, X_trimmed_imputed = _raw_dataset_4_tsfel(df_raw)
    cfg = tsfel.get_features_by_domain()

    # Eliminación de feature irrelevante (human_range_energy)
    if 'human_range_energy' in cfg['spectral']:
        del cfg['spectral']['human_range_energy']

    # Parámetros
    wavelet_feats = ['wavelet_abs_mean', 'wavelet_energy', 'wavelet_entropy', 'wavelet_std', 'wavelet_var']
    for domain in cfg.keys():
        for feature, config in cfg[domain].items():
            if feature in wavelet_feats:
                config['parameters']['max_width'] = 30
            elif feature == 'spectrogram_mean_coeff':
                config['parameters']['bins'] = 64
            elif feature == 'neighbourhood_peaks':
                config['parameters']['n'] = 14
            elif feature == 'ecdf_percentile':
                config['parameters']['percentile'] = [0.1, 0.9]
            elif feature == 'ecdf_slope':
                config['parameters']['p_init'] = 0.5
                config['parameters']['p_end'] = 0.85
            elif feature == 'hist_mode':
                config['parameters']['nbins'] = 20
            elif feature == 'mfcc':
                config['parameters']['pre_emphasis'] = 0
                config['parameters']['nfft'] = 128
                config['parameters']['nfilt'] = 20
                config['parameters']['num_ceps'] = 8
            elif feature == 'lpcc':
                config['parameters']['n_coeff'] = 8

    # Extracción
    tsfel_features_list = []
    desc_msg = "TSFEL: extracting features..."
    for arr in tqdm(X_trimmed_imputed, desc=desc_msg):
        feats = calc_window_features(cfg, arr, fs=1)
        # Con dedup_columns, se corrige el bug que asigna mismos nombres a columnas similares
        # (ej: coefficient_0.41Hz == coefficient_0.45Hz) 
        feats.columns = utils.dedup_columns(list(feats.columns))
        tsfel_features_list.append(feats)
    df_tsfel = pd.concat(tsfel_features_list, ignore_index=True)

    # Constantes y correlación eliminadas
    df_tsfel = df_tsfel.loc[:, df_tsfel.nunique() > 1]
    print(f"\nFeatures after removing constants: {df_tsfel.shape[1]}")
    # (por Pearson, umbral 0.95 por defecto)
    dropped_features, df_tsfel_reduced = correlated_features(df_tsfel, drop_correlated=True)
    print(f"Removed {len(dropped_features)} highly correlated features.")
    print(f"Final TSFEL features count: {df_tsfel_reduced.shape[1]}")
    # Suma total de nulos
    print(f"Total missing values: {df_tsfel_reduced.isna().sum().sum()} | Filling with 0 before saving")
    df_tsfel_reduced_nulls = df_tsfel_reduced.fillna(0)

    # Índices, variable objetivo y guardado según max_gap_threshold
    df_tsfel_reduced_nulls.index = df_670.index
    df_tsfel_reduced_nulls['FLAG'] = df_670['FLAG']
    os.makedirs(output_path, exist_ok=True)

    # Guardado para gap 670
    path_670 = os.path.join(output_path, 'tsfel_features_gap670.csv')
    df_tsfel_reduced_nulls.to_csv(path_670, index=False)
    print(" * Succesfully created & saved:", path_670)
    
    # Guardado para gap 250
    df_250 = utils.drop_large_max_gaps(df_670, max_gap_threshold=250)
    df_tsfel_250 = df_tsfel_reduced_nulls.loc[df_250.index]
    path_250 = os.path.join(output_path, 'tsfel_features_gap250.csv')
    df_tsfel_250.to_csv(path_250, index=False)
    print(" * Succesfully created & saved:", path_250)


def _drop_correlated(df, threshold=0.95, target_col='FLAG'):
    """
    Remove highly correlated features by excluding the target variable.
    """
    # Separa la objetivo
    if target_col in df.columns:
        y = df[target_col]
        df_features = df.drop(columns=[target_col])
    else:
        y = None
        df_features = df.copy()
    # Filtro de correlación con conteo de eliminaciones
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_reduced = df_features.drop(columns=to_drop)
    # Unión con la variable objetivo
    if y is not None:
        df_reduced[target_col] = y
        
    return df_reduced, len(to_drop)


def _add_isolation_forest(df, target_col='FLAG', train_size=0.70, random_state=SEED):
    """
    Add Isolation Forest scores by fitting only on the training subset.
    """
    # Solo sobre train
    train_df, _ = train_test_split(
        df, train_size=train_size, stratify=df[target_col], random_state=random_state
    )
    
    # Se evita la objetivo y se obtiene la tasa de contaminación (fraude)
    features = [col for col in df.columns if col != target_col]
    contamination_rate = train_df[target_col].mean()
    # Se entrena
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination_rate,
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(train_df[features])
    
    # Y se aplica a todo el conjunto
    df_with_if = df.copy()
    df_with_if['IF_score'] = iso.decision_function(df_with_if[features])
    
    return df_with_if

def assemble_feature_datasets(df_raw, windows=[2, 4, 8, 12], seed=SEED, 
                                    tsfel_input_dataset_larger_gap_path='data/tsfel_datasets/tsfel_features_gap670.csv', 
                                    output_path='data/all_features/'):
    """
    Generate all datasets by combining own extractions and TSFEL.
    Dynamically generate combinations for max gap (670 and 250) and n_windows,
    using a single unified TSFEL extraction.
    """
    os.makedirs(output_path, exist_ok=True)

    print("Preprocessing raw dataset...")
    # A partir del gap más permisivo
    df_670 = (df_raw
        .pipe(utils.drop_industrial_client)
        .pipe(utils.drop_clientID)
        .pipe(utils.drop_duplicated_rows)
        .pipe(utils.drop_inactive_clients)
        .pipe(utils.drop_n_not_consecutive_values, n=MIN_CONSECUTIVE_DAYS)
        .pipe(utils.drop_large_max_gaps, max_gap_threshold=670)
    )
    
    # Se extrae evitando dependencia de n_windows (extract_consumption_features)
    print("Extracting features...")
    df_entry = extract_entry_and_lifespan(df_670)
    X_trimmed = utils.get_trimmed_series(df_670)
    df_zeros = extract_zero_features(X_trimmed)
    df_nulls = extract_null_features(X_trimmed)
    df_no_errors = utils.interpolate_spikes(df_670)
    X_trimmed_no_errors = utils.get_trimmed_series(df_no_errors)
    df_seasonal = extract_seasonal_features(df_no_errors)

    # Índices para la unión
    df_entry.index = df_670.index
    df_zeros.index = df_670.index
    df_nulls.index = df_670.index
    df_seasonal.index = df_670.index

    # Carga TSFEL
    df_tsfel_670 = pd.read_csv(tsfel_input_dataset_larger_gap_path)
    df_tsfel_670.index = df_670.index
    
    # Referencias y subset para gap 250
    df_250_reference = utils.drop_large_max_gaps(df_670, max_gap_threshold=250)
    indices_250 = df_250_reference.index
    df_tsfel_250 = df_tsfel_670.loc[indices_250]

    # Cálculo por partición o ventana, unión y guardado
    for n_win in windows:          
        df_consumption = extract_consumption_features(X_trimmed_no_errors, n_windows=n_win)
        df_consumption.index = df_670.index

        # Para gap 670
        df_final_670 = pd.concat([
            df_entry, df_zeros, df_nulls, df_seasonal, df_consumption, df_tsfel_670
        ], axis=1)
        df_final_670, dropped_670 = _drop_correlated(df_final_670, threshold=0.95)    
        # Aplicando Isolation Forest
        df_final_670 = _add_isolation_forest(df_final_670, random_state=seed)
        out_670 = os.path.join(output_path, f'gap670_windows{n_win}.csv')
        df_final_670.to_csv(out_670, index=False)
        
        # Para gap 250
        df_final_250 = pd.concat([
            df_entry.loc[indices_250], 
            df_zeros.loc[indices_250], 
            df_nulls.loc[indices_250], 
            df_seasonal.loc[indices_250], 
            df_consumption.loc[indices_250], 
            df_tsfel_250
        ], axis=1)
        df_final_250, dropped_250 = _drop_correlated(df_final_250, threshold=0.95)
        df_final_250 = _add_isolation_forest(df_final_250, random_state=seed)
        out_250 = os.path.join(output_path, f'gap250_windows{n_win}.csv')
        df_final_250.to_csv(out_250, index=False)
        
        # Logs
        print(f"\n  Successfully created & saved ({n_win} windows):")
        print(f"       - {out_670}")
        print(f"         * Total features: {df_final_670.shape[1]} | Dropped {dropped_670} due to High Correlation")
        print(f"       - {out_250}")
        print(f"         * Total features: {df_final_250.shape[1]} | Dropped {dropped_250} due to High Correlation")