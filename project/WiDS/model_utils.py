"""
模型工具函数 - 用于 Streamlit 应用
功能：提供模型训练时使用的特征准备函数
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_features(train_df_filled, fill_missing=True, standardize=True):
    """
    准备特征数据：处理缺失值、编码分类特征、标准化
    
    Args:
        train_df_filled: 预处理后的训练数据DataFrame
        fill_missing: 是否填充数值特征缺失值（True=填充，False=保留缺失值）
        standardize: 是否进行特征标准化（True=标准化，False=不标准化）
    
    Returns:
        X: 特征矩阵
        y: 目标变量（如果存在）
        feature_names: 特征名称列表
        preprocessor: 预处理器字典（包含scaler和encoders）
    """
    # 复制数据以避免修改原始数据
    df = train_df_filled.copy()
    
    # 分离目标变量和标识符
    target_col = 'hospital_death'
    id_cols = ['encounter_id', 'patient_id', 'hospital_id']
    
    # 提取目标变量（如果存在）
    y = None
    if target_col in df.columns:
        y = df[target_col].values
    
    # 移除目标变量和标识符
    df_features = df.drop(columns=[target_col] + [col for col in id_cols if col in df.columns], errors='ignore')
    
    # 移除APACHE死亡概率特征（这些特征可能包含目标变量信息，属于数据泄露）
    apache_prob_features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
    features_to_remove = [f for f in apache_prob_features if f in df_features.columns]
    if features_to_remove:
        df_features = df_features.drop(columns=features_to_remove)
    
    # 处理分类特征
    object_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in object_cols:
        if col in df_features.columns:
            # 使用LabelEncoder进行编码
            le = LabelEncoder()
            # 处理缺失值：用'Missing'填充（分类特征必须填充才能编码）
            df_features[col] = df_features[col].fillna('Missing')
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            encoders[col] = le
    
    # 处理数值特征缺失值
    if fill_missing:
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df_features[col].isnull().any():
                median_val = df_features[col].median()
                df_features[col].fillna(median_val, inplace=True)
    
    # 特征标准化
    scaler = None
    if standardize:
        from sklearn.preprocessing import StandardScaler
        # 注意：StandardScaler不支持缺失值，如果保留缺失值则不标准化
        if not fill_missing and df_features.select_dtypes(include=[np.number]).isnull().any().any():
            scaler = None
            X = df_features.copy()
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_features)
            X = pd.DataFrame(X_scaled, columns=df_features.columns, index=df_features.index)
    else:
        X = df_features.copy()
    
    # 保存预处理器
    preprocessor = {
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': df_features.columns.tolist(),
        'fill_missing': fill_missing,
        'standardize': standardize
    }
    
    return X, y, df_features.columns.tolist(), preprocessor
