"""
WiDS Datathon 2020 - 特征工程模块
功能：创建新的有意义的特征，提升模型性能
作者：资深医疗数据科学家
日期：2024
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def create_gcs_features(df):
    """
    创建GCS（格拉斯哥昏迷评分）相关特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了GCS特征的DataFrame
    """
    print("创建GCS特征...")
    
    # GCS总分 = 眼睛 + 运动 + 语言
    if all(col in df.columns for col in ['gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache']):
        # 处理缺失值：如果任一组件缺失，总分也缺失
        gcs_total = df['gcs_eyes_apache'] + df['gcs_motor_apache'] + df['gcs_verbal_apache']
        # 如果gcs_unable_apache=1，表示无法评估，设为缺失
        if 'gcs_unable_apache' in df.columns:
            gcs_total[df['gcs_unable_apache'] == 1] = np.nan
        df['gcs_total'] = gcs_total
        print(f"  ✓ 创建 gcs_total: 范围 [{df['gcs_total'].min():.1f}, {df['gcs_total'].max():.1f}]")
    
    # GCS运动评分（通常是最重要的）
    if 'gcs_motor_apache' in df.columns:
        df['gcs_motor_important'] = df['gcs_motor_apache']
        print(f"  ✓ 保留 gcs_motor_apache 作为重要特征")
    
    return df

def create_vital_signs_features(df):
    """
    创建生命体征相关特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了生命体征特征的DataFrame
    """
    print("创建生命体征特征...")
    
    # 1. 血压相关特征
    # 收缩压范围（最大值-最小值）
    if all(col in df.columns for col in ['d1_sysbp_max', 'd1_sysbp_min']):
        df['d1_sysbp_range'] = df['d1_sysbp_max'] - df['d1_sysbp_min']
        print(f"  ✓ 创建 d1_sysbp_range")
    
    # 舒张压范围
    if all(col in df.columns for col in ['d1_diasbp_max', 'd1_diasbp_min']):
        df['d1_diasbp_range'] = df['d1_diasbp_max'] - df['d1_diasbp_min']
        print(f"  ✓ 创建 d1_diasbp_range")
    
    # 平均动脉压范围
    if all(col in df.columns for col in ['d1_mbp_max', 'd1_mbp_min']):
        df['d1_mbp_range'] = df['d1_mbp_max'] - df['d1_mbp_min']
        print(f"  ✓ 创建 d1_mbp_range")
    
    # 2. 心率相关特征
    if all(col in df.columns for col in ['d1_heartrate_max', 'd1_heartrate_min']):
        df['d1_heartrate_range'] = df['d1_heartrate_max'] - df['d1_heartrate_min']
        df['d1_heartrate_mean'] = (df['d1_heartrate_max'] + df['d1_heartrate_min']) / 2
        print(f"  ✓ 创建 d1_heartrate_range 和 d1_heartrate_mean")
    
    # 3. 血氧饱和度相关特征
    if all(col in df.columns for col in ['d1_spo2_max', 'd1_spo2_min']):
        df['d1_spo2_range'] = df['d1_spo2_max'] - df['d1_spo2_min']
        df['d1_spo2_mean'] = (df['d1_spo2_max'] + df['d1_spo2_min']) / 2
        print(f"  ✓ 创建 d1_spo2_range 和 d1_spo2_mean")
    
    # 4. 体温相关特征
    if all(col in df.columns for col in ['d1_temp_max', 'd1_temp_min']):
        df['d1_temp_range'] = df['d1_temp_max'] - df['d1_temp_min']
        df['d1_temp_mean'] = (df['d1_temp_max'] + df['d1_temp_min']) / 2
        print(f"  ✓ 创建 d1_temp_range 和 d1_temp_mean")
    
    # 5. 呼吸频率相关特征
    if all(col in df.columns for col in ['d1_resprate_max', 'd1_resprate_min']):
        df['d1_resprate_range'] = df['d1_resprate_max'] - df['d1_resprate_min']
        df['d1_resprate_mean'] = (df['d1_resprate_max'] + df['d1_resprate_min']) / 2
        print(f"  ✓ 创建 d1_resprate_range 和 d1_resprate_mean")
    
    return df

def create_lab_features(df):
    """
    创建实验室检查相关特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了实验室特征的DataFrame
    """
    print("创建实验室检查特征...")
    
    # 1. 肾功能指标
    if all(col in df.columns for col in ['d1_creatinine_max', 'd1_creatinine_min']):
        df['d1_creatinine_range'] = df['d1_creatinine_max'] - df['d1_creatinine_min']
        print(f"  ✓ 创建 d1_creatinine_range")
    
    if all(col in df.columns for col in ['d1_bun_max', 'd1_bun_min']):
        df['d1_bun_range'] = df['d1_bun_max'] - df['d1_bun_min']
        print(f"  ✓ 创建 d1_bun_range")
    
    # 2. 肝功能指标
    if all(col in df.columns for col in ['d1_bilirubin_max', 'd1_bilirubin_min']):
        df['d1_bilirubin_range'] = df['d1_bilirubin_max'] - df['d1_bilirubin_min']
        print(f"  ✓ 创建 d1_bilirubin_range")
    
    if all(col in df.columns for col in ['d1_albumin_max', 'd1_albumin_min']):
        df['d1_albumin_range'] = df['d1_albumin_max'] - df['d1_albumin_min']
        print(f"  ✓ 创建 d1_albumin_range")
    
    # 3. 血液指标
    if all(col in df.columns for col in ['d1_hematocrit_max', 'd1_hematocrit_min']):
        df['d1_hematocrit_range'] = df['d1_hematocrit_max'] - df['d1_hematocrit_min']
        print(f"  ✓ 创建 d1_hematocrit_range")
    
    if all(col in df.columns for col in ['d1_wbc_max', 'd1_wbc_min']):
        df['d1_wbc_range'] = df['d1_wbc_max'] - df['d1_wbc_min']
        print(f"  ✓ 创建 d1_wbc_range")
    
    # 4. 电解质
    if all(col in df.columns for col in ['d1_sodium_max', 'd1_sodium_min']):
        df['d1_sodium_range'] = df['d1_sodium_max'] - df['d1_sodium_min']
        print(f"  ✓ 创建 d1_sodium_range")
    
    if all(col in df.columns for col in ['d1_potassium_max', 'd1_potassium_min']):
        df['d1_potassium_range'] = df['d1_potassium_max'] - df['d1_potassium_min']
        print(f"  ✓ 创建 d1_potassium_range")
    
    return df

def create_temporal_features(df):
    """
    创建时间序列相关特征（d1 vs h1的变化）
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了时间序列特征的DataFrame
    """
    print("创建时间序列特征...")
    
    # 1. 生命体征变化（d1到h1）
    vital_pairs = [
        ('d1_heartrate', 'h1_heartrate'),
        ('d1_sysbp', 'h1_sysbp'),
        ('d1_mbp', 'h1_mbp'),
        ('d1_resprate', 'h1_resprate'),
        ('d1_spo2', 'h1_spo2'),
        ('d1_temp', 'h1_temp')
    ]
    
    for d1_prefix, h1_prefix in vital_pairs:
        max_col = f'{d1_prefix}_max'
        h1_max_col = f'{h1_prefix}_max'
        if max_col in df.columns and h1_max_col in df.columns:
            df[f'{d1_prefix}_to_{h1_prefix}_change'] = df[h1_max_col] - df[max_col]
            print(f"  ✓ 创建 {d1_prefix}_to_{h1_prefix}_change")
    
    # 2. 实验室指标变化
    lab_pairs = [
        ('d1_creatinine', 'h1_creatinine'),
        ('d1_bun', 'h1_bun'),
        ('d1_bilirubin', 'h1_bilirubin'),
        ('d1_albumin', 'h1_albumin'),
        ('d1_wbc', 'h1_wbc')
    ]
    
    for d1_prefix, h1_prefix in lab_pairs:
        max_col = f'{d1_prefix}_max'
        h1_max_col = f'{h1_prefix}_max'
        if max_col in df.columns and h1_max_col in df.columns:
            df[f'{d1_prefix}_to_{h1_prefix}_change'] = df[h1_max_col] - df[max_col]
            print(f"  ✓ 创建 {d1_prefix}_to_{h1_prefix}_change")
    
    return df

def create_missing_pattern_features(df):
    """
    创建缺失值模式特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了缺失值模式特征的DataFrame
    """
    print("创建缺失值模式特征...")
    
    # 关键特征列表
    key_features = [
        'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache',
        'd1_sysbp_min', 'd1_mbp_min', 'd1_heartrate_min',
        'd1_spo2_min', 'd1_temp_min', 'd1_resprate_min',
        'd1_creatinine_min', 'd1_bun_max', 'd1_albumin_min'
    ]
    
    # 计算关键特征缺失数量
    available_key_features = [f for f in key_features if f in df.columns]
    if available_key_features:
        df['key_features_missing_count'] = df[available_key_features].isnull().sum(axis=1)
        print(f"  ✓ 创建 key_features_missing_count (基于 {len(available_key_features)} 个关键特征)")
    
    # 计算所有数值特征的缺失比例
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df['numeric_features_missing_ratio'] = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols)
        print(f"  ✓ 创建 numeric_features_missing_ratio")
    
    return df

def create_interaction_features(df):
    """
    创建交互特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了交互特征的DataFrame
    """
    print("创建交互特征...")
    
    # 1. 年龄与BMI交互
    if all(col in df.columns for col in ['age', 'bmi']):
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        print(f"  ✓ 创建 age_bmi_interaction")
    
    # 2. 年龄与GCS交互
    if 'age' in df.columns and 'gcs_total' in df.columns:
        df['age_gcs_interaction'] = df['age'] * df['gcs_total']
        print(f"  ✓ 创建 age_gcs_interaction")
    
    # 3. 血压与心率交互
    if all(col in df.columns for col in ['d1_mbp_mean', 'd1_heartrate_mean']):
        df['mbp_hr_interaction'] = df['d1_mbp_mean'] * df['d1_heartrate_mean']
        print(f"  ✓ 创建 mbp_hr_interaction")
    
    return df

def apply_feature_engineering(df):
    """
    应用所有特征工程
    
    Args:
        df: 原始数据DataFrame
    
    Returns:
        df: 添加了新特征的DataFrame
    """
    print("=" * 80)
    print("特征工程")
    print("=" * 80)
    print()
    
    original_shape = df.shape
    print(f"原始数据形状: {original_shape[0]:,} 行 × {original_shape[1]} 列")
    print()
    
    # 应用各种特征工程
    df = create_gcs_features(df)
    print()
    
    df = create_vital_signs_features(df)
    print()
    
    df = create_lab_features(df)
    print()
    
    df = create_temporal_features(df)
    print()
    
    df = create_missing_pattern_features(df)
    print()
    
    df = create_interaction_features(df)
    print()
    
    new_shape = df.shape
    print("=" * 80)
    print(f"特征工程完成")
    print(f"原始特征数: {original_shape[1]}")
    print(f"新特征数: {new_shape[1]}")
    print(f"新增特征数: {new_shape[1] - original_shape[1]}")
    print("=" * 80)
    print()
    
    return df
