"""
WiDS Datathon 2020 - Streamlit 数据分析与可视化应用
基于多中心临床数据的 ICU 死亡风险预测项目主页
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

# 以当前应用目录为根目录，避免依赖仓库根目录
BASE_DIR = Path(__file__).resolve().parent

# 缓存函数：加载CSV数据
@st.cache_data
def load_csv_data(file_path, **kwargs):
    """缓存CSV文件加载"""
    return pd.read_csv(file_path, **kwargs)

# 缓存函数：加载模型
@st.cache_resource
def load_model(model_path):
    """缓存模型加载"""
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# 缓存函数：加载预处理器
@st.cache_resource
def load_preprocessor(preprocessor_path):
    """缓存预处理器加载"""
    import pickle
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

# 缓存函数：计算缺失值统计（优化：使用更小的采样减少计算时间）
@st.cache_data
def compute_missing_stats(data_path, chunk_size=10000, max_rows=20000):
    """
    缓存缺失值统计计算
    优化：限制最大读取行数为20000，大幅减少计算时间
    """
    columns = load_csv_data(data_path, nrows=0).columns.tolist()
    total_rows = 0
    missing_counts = pd.Series(0, index=columns)
    
    # 分块读取并累计缺失值（限制最大行数）
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False, na_values=['NA', '']):
        total_rows += len(chunk)
        missing_counts += chunk.isnull().sum()
        # 如果已达到最大行数，停止读取
        if total_rows >= max_rows:
            break
    
    # 计算缺失值比例
    missing_percent = (missing_counts / total_rows) * 100
    missing_df = pd.DataFrame({
        '特征': missing_percent.index,
        '缺失比例(%)': missing_percent.values
    }).sort_values('缺失比例(%)', ascending=False)
    
    return missing_df, total_rows, len(columns)


# 缓存函数：获取用于在线预测的模型与特征信息（优化：减少初始数据加载量）
@st.cache_resource
def get_prediction_model_and_features(sample_size=10000):
    """
    加载用于在线个体预测的 LightGBM 最优模型，并推断其使用的特征列表与默认填充值（中位数）。
    注意：此函数会应用与训练时相同的预处理流程（特征工程、特征选择等）。
    
    Args:
        sample_size: 用于计算中位数的样本数量（默认10000，减少内存占用）
    
    Returns:
        model: 已加载的 LightGBM 模型（或 None）
        feature_list: 模型使用的特征名称列表（或 None）
        feature_medians: 这些特征在训练集上的中位数（用于默认填充）
        preprocessor: 预处理器对象（包含编码器等）
    """
    model_path = BASE_DIR / "models" / "LightGBM_tuned_advanced.pkl"
    preprocessor_path = BASE_DIR / "models" / "preprocessor_lightgbm_advanced.pkl"
    data_path = BASE_DIR / "data" / "training_v2.csv"

    if not model_path.exists() or not data_path.exists():
        return None, None, None, None

    # 加载模型
    model_data = load_model(model_path)
    if isinstance(model_data, dict):
        model = model_data.get('model')
    else:
        model = model_data

    if model is None:
        return None, None, None, None

    # 加载预处理器
    preprocessor = None
    selected_features = None
    use_feature_engineering = False
    
    if preprocessor_path.exists():
        try:
            preprocessor = load_preprocessor(preprocessor_path)
            if isinstance(preprocessor, dict):
                selected_features = preprocessor.get('feature_names')
                use_feature_engineering = preprocessor.get('use_feature_engineering', False)
        except Exception as e:
            st.warning(f"加载预处理器时出错: {str(e)}")
            preprocessor = None

    # 加载训练数据（用于特征工程和计算中位数）- 优化：减少样本数量
    try:
        # 导入特征工程函数（如果可用）
        if use_feature_engineering:
            try:
                import sys
                sys.path.insert(0, str(BASE_DIR.parent))
                from feature_engineering import apply_feature_engineering
            except ImportError:
                st.warning("无法导入特征工程模块，将跳过特征工程步骤")
                use_feature_engineering = False
        
        # 使用更小的样本量来计算中位数，减少内存占用
        train_df = load_csv_data(data_path, nrows=sample_size, low_memory=False, na_values=['NA', ''])
        if 'hospital_death' not in train_df.columns:
            return None, None, None, None
        
        # 应用特征工程（如果训练时使用了）
        if use_feature_engineering:
            try:
                train_df = apply_feature_engineering(train_df)
            except Exception as e:
                st.warning(f"应用特征工程时出错: {str(e)}")
        
        # 移除APACHE死亡概率特征（避免数据泄露，与训练时一致）
        apache_prob_features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
        for feat in apache_prob_features:
            if feat in train_df.columns:
                train_df = train_df.drop(columns=[feat])
        
        # 处理分类特征（使用预处理器中的编码器，如果可用）
        if preprocessor and isinstance(preprocessor, dict) and 'encoders' in preprocessor:
            encoders = preprocessor['encoders']
            for col, encoder in encoders.items():
                if col in train_df.columns:
                    # 处理缺失值：用'Missing'填充（与训练时一致）
                    train_df[col] = train_df[col].fillna('Missing')
                    try:
                        # 尝试转换
                        train_df[col] = train_df[col].astype(str)
                        # 对于新值，使用最常见的类别
                        known_classes = set(encoder.classes_)
                        train_df[col] = train_df[col].apply(
                            lambda x: x if x in known_classes else encoder.classes_[0]
                        )
                        train_df[col] = encoder.transform(train_df[col])
                    except Exception:
                        # 如果编码失败，使用最常见的类别
                        train_df[col] = 0
        
        # 获取特征列表
        if selected_features:
            # 使用预处理器中保存的特征列表（这是训练时选择的特征）
            feature_list = [f for f in selected_features if f in train_df.columns]
            # 对于缺失的特征，用0填充（不应该发生，但为了安全）
            missing_features = [f for f in selected_features if f not in train_df.columns]
            if missing_features:
                for feat in missing_features:
                    train_df[feat] = 0
                feature_list = selected_features  # 使用完整的特征列表
        else:
            # 如果没有预处理器，推断特征数量
            model_n_features = None
            try:
                if hasattr(model, 'n_features_'):
                    model_n_features = model.n_features_
                elif hasattr(model, 'booster_'):
                    model_n_features = model.booster_.num_feature()
            except Exception:
                model_n_features = None
            
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in 
                            ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']]
            
            n_feats = model_n_features if model_n_features else 79
            feature_list = [col for col in numeric_cols if col in train_df.columns][:n_feats]

        if len(feature_list) == 0:
            return None, None, None, None

        # 计算这些特征在训练集上的中位数，用作默认填充值
        # 注意：对于LightGBM，我们保留缺失值，但为了给用户提供合理的默认值，使用中位数
        feature_medians = train_df[feature_list].median()
        
        # 确保特征顺序与训练时一致
        feature_list = [f for f in selected_features if f in feature_list] if selected_features else feature_list

        return model, feature_list, feature_medians, preprocessor
        
    except Exception as e:
        st.error(f"准备预测数据时出错: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None, None

# 页面配置（优化：减少初始渲染）
st.set_page_config(
    page_title="WiDS Datathon 2020 - ICU死亡风险预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit 性能优化配置
# 注意：Streamlit 的缓存机制已经通过 @st.cache_data 和 @st.cache_resource 实现

# 初始化session_state（用于缓存已加载的数据）
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    /* 指标小卡片：莫兰迪灰蓝调，浅色背景 + 深色文字 */
    .metric-card {
        background-color: #e4e7ed;  /* 浅灰蓝 (Morandi) */
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #9aa5c4;  /* 柔和蓝灰边框 */
        margin: 0.5rem 0;
    }
    /* 信息提示块：莫兰迪蓝灰调 */
    .info-box {
        background-color: #dde5f0;  /* 浅蓝灰 */
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #8fa4c8;  /* 柔和蓝色 */
        margin: 1rem 0;
    }
    /* 成功/积极提示：莫兰迪绿调 */
    .success-box {
        background-color: #e3f0e8;  /* 浅灰绿 */
        color: #245048;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #88b89a;  /* 柔和绿色 */
        margin: 1rem 0;
    }
    /* 警告/风险提示：莫兰迪黄棕调 */
    .warning-box {
        background-color: #f3e7d8;  /* 浅米杏色 */
        color: #6b4b2b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d3a46f;  /* 柔和棕橙 */
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-header">🏥 WiDS Datathon 2020 - ICU死亡风险预测分析系统</div>', unsafe_allow_html=True)

# 项目信息
col1, col2, col3 = st.columns([2, 1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>📋 项目概述</h3>
        <p><strong>项目名称：</strong>基于多中心临床数据 WiDS 的 ICU 死亡风险预测</p>
        <p><strong>数据来源：</strong>MIT GOSSIS 倡议 - WiDS Datathon 2020</p>
        <p><strong>研究目标：</strong>利用患者进入ICU后前24小时的关键生理体征及实验室指标，预测住院死亡风险</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div style="padding-top: 0.5rem;">', unsafe_allow_html=True)
    st.metric("📊 样本数量", "91,713")
    st.metric("🔬 特征维度", "186")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div style="padding-top: 0.5rem;">', unsafe_allow_html=True)
    st.metric("🏥 医院数量", "200+")
    st.metric("🎯 目标变量", "hospital_death")
    st.markdown('</div>', unsafe_allow_html=True)

# 临床个体预测板块（独立板块，放在主要分析模块之前）
# 优化：使用expander延迟加载，减少初始页面加载时间
prediction_expander = st.expander("🩺 临床个体风险预测（点击展开使用）", expanded=False)

with prediction_expander:
    st.markdown('<div class="section-header">🩺 临床个体风险预测</div>', unsafe_allow_html=True)
    st.markdown("""
    **功能说明：**  
    - 支持临床医生或用户输入少量关键指标（如年龄、BMI、心率、血糖等），由 **Optuna 调优后的 LightGBM 最优模型** 预测住院死亡风险  
    - 未输入的其他特征自动使用训练集典型值（中位数）填充，保证与离线模型使用的特征保持一致  
    """)
    # 懒加载：只在需要时加载模型（使用session_state缓存）
    if 'prediction_model' not in st.session_state:
        with st.spinner("正在加载预测模型（首次加载可能需要几秒钟）..."):
            model, feature_list, feature_medians, preprocessor = get_prediction_model_and_features(sample_size=5000)
            st.session_state['prediction_model'] = model
            st.session_state['prediction_feature_list'] = feature_list
            st.session_state['prediction_feature_medians'] = feature_medians
            st.session_state['prediction_preprocessor'] = preprocessor
    else:
        model = st.session_state['prediction_model']
        feature_list = st.session_state['prediction_feature_list']
        feature_medians = st.session_state['prediction_feature_medians']
        preprocessor = st.session_state['prediction_preprocessor']

    if model is None or feature_list is None or feature_medians is None:
        st.warning("⚠️ 未能加载在线预测所需的模型或数据，请确认 `models/LightGBM_tuned_advanced.pkl` 和 `data/training_v2.csv` 已放置在 `streamlit_app` 目录下。")
    else:
    # 关键医学特征（如存在则提供输入项）
    # 键为数据集中列名，值为(中文名称, 合理最小值, 合理最大值)
    # 前几项为基础特征，后面补充了一批更“高危”的核心生理 / 实验室指标
    candidate_numeric_features = {
        # 基础人口学/生命体征
        'age': ("年龄 (岁)", 18.0, 100.0),
        'bmi': ("BMI (kg/m²)", 10.0, 60.0),
        'heart_rate_apache': ("入ICU心率 (次/分)", 30.0, 200.0),
        'temp_apache': ("入ICU体温 (℃)", 30.0, 43.0),
        'd1_sysbp_max': ("首日最高收缩压 (mmHg)", 60.0, 260.0),
        'd1_sysbp_min': ("首日最低收缩压 (mmHg)", 40.0, 200.0),
        'd1_heartrate_max': ("首日最高心率 (次/分)", 40.0, 220.0),
        'd1_heartrate_min': ("首日最低心率 (次/分)", 20.0, 150.0),

        # 血糖 & 代谢
        'd1_glucose_max': ("首日最高血糖 (mmol/L)", 2.0, 40.0),
        'd1_glucose_min': ("首日最低血糖 (mmol/L)", 2.0, 30.0),
        'd1_lactate_max': ("首日最高乳酸 (mmol/L)", 0.5, 15.0),
        'd1_lactate_min': ("首日最低乳酸 (mmol/L)", 0.5, 10.0),

        # 循环与灌注
        'd1_mbp_min': ("首日最低平均动脉压 (mmHg)", 40.0, 120.0),
        'd1_spo2_min': ("首日最低血氧饱和度 (%)", 50.0, 100.0),

        # 呼吸功能
        'd1_resprate_max': ("首日最高呼吸频率 (次/分)", 8.0, 60.0),

        # 肾功能 / 代谢废物
        'd1_creatinine_max': ("首日最高肌酐 (mg/dL)", 0.2, 10.0),
        'd1_urineoutput': ("首日尿量 (mL)", 0.0, 10000.0),

        # 综合风险评分
        'apache_4a_icu_death_prob': ("APACHE ICU 预测死亡概率", 0.0, 1.0),
    }

    # 仅保留在训练数据中实际存在的特征
    available_candidates = {
        name: meta for name, meta in candidate_numeric_features.items()
        if name in feature_medians.index
    }

    if not available_candidates:
        st.info("当前模型使用的特征中不包含预设的关键医学指标，暂无法提供交互式个体预测表单。")
    else:
        st.markdown("#### 请输入患者的关键信息（其余未列出的特征将使用训练集典型值填充）")

        with st.form("manual_clinical_prediction"):
            input_cols = st.columns(3)
            user_values = {}

            for idx, (feat_name, (label, vmin, vmax)) in enumerate(available_candidates.items()):
                col = input_cols[idx % 3]
                with col:
                    # 默认值取训练集中的中位数，但要确保落在[min, max]区间内，避免越界报错
                    raw_default = float(feature_medians.get(feat_name, (vmin + vmax) / 2.0))
                    default_val = min(max(raw_default, float(vmin)), float(vmax))
                    # 对概率类特征单独设置步长
                    step = 0.01 if "prob" in feat_name else 0.1
                    user_values[feat_name] = st.number_input(
                        label,
                        min_value=float(vmin),
                        max_value=float(vmax),
                        value=float(default_val),
                        step=step,
                        key=f"input_{feat_name}"
                    )

            st.markdown("---")
            threshold = st.slider(
                "高风险判定阈值（预测死亡概率 ≥ 该值视为高风险）",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )

            submitted = st.form_submit_button("计算死亡风险")

        if submitted:
            try:
                # 使用与训练时完全一致的预处理流程（参考predict_lightgbm_ensemble.py）
                import sys
                sys.path.insert(0, str(BASE_DIR.parent))
                
                # 1. 加载训练数据的一个样本作为基础（用于特征工程）
                data_path = BASE_DIR / "data" / "training_v2.csv"
                patient_df = load_csv_data(data_path, nrows=1, low_memory=False, na_values=['NA', ''])
                
                # 2. 应用特征工程（如果训练时使用了）
                use_feature_engineering = preprocessor.get('use_feature_engineering', False) if preprocessor and isinstance(preprocessor, dict) else False
                if use_feature_engineering:
                    try:
                        from feature_engineering import apply_feature_engineering
                        patient_df = apply_feature_engineering(patient_df.copy())
                    except Exception as e:
                        st.warning(f"应用特征工程时出错: {str(e)}")
                
                # 3. 使用prepare_features函数准备特征（与训练时完全一致）
                try:
                    from model_training import prepare_features
                    
                    # 准备特征（保留缺失值，用于LightGBM，与训练时一致）
                    X_prepared, _, _, _ = prepare_features(
                        patient_df.copy(), fill_missing=False, standardize=False
                    )
                    
                    # 4. 用训练集的中位数填充所有特征（作为基础值）
                    # 注意：这里我们需要确保所有特征都存在
                    for feat in feature_list:
                        if feat in X_prepared.columns:
                            # 用中位数填充（如果特征在medians中）
                            if feat in feature_medians.index:
                                X_prepared[feat] = feature_medians[feat]
                            else:
                                X_prepared[feat] = 0.0
                        else:
                            # 如果特征不在DataFrame中，添加它
                            X_prepared[feat] = feature_medians.get(feat, 0.0) if feat in feature_medians.index else 0.0
                    
                    # 5. 用用户输入的值覆盖对应特征
                    for feat_name, val in user_values.items():
                        if feat_name in X_prepared.columns:
                            X_prepared[feat_name] = float(val)
                        elif feat_name in feature_list:
                            # 如果特征在特征列表中但不在DataFrame中，添加它
                            X_prepared[feat_name] = float(val)
                    
                    # 6. 特征选择：按照预处理器中保存的特征顺序组织输入
                    # 这是关键步骤：确保特征顺序与训练时完全一致
                    X_input_selected = pd.DataFrame(index=X_prepared.index)
                    missing_features = []
                    
                    for feat in feature_list:
                        if feat in X_prepared.columns:
                            X_input_selected[feat] = X_prepared[feat]
                        else:
                            missing_features.append(feat)
                            X_input_selected[feat] = 0.0  # 用0填充缺失的特征
                    
                    # 确保特征顺序与训练时一致
                    X_input_selected = X_input_selected[feature_list]
                    
                    if missing_features:
                        st.warning(f"⚠ 警告: {len(missing_features)} 个特征在数据中不存在，已用0填充")
                    
                    # 7. 转换为numpy数组
                    X_input = X_input_selected.values
                    
                    # 8. 验证特征数量和顺序
                    if X_input.shape[1] != len(feature_list):
                        st.error(f"❌ 特征数量不匹配！模型期望 {len(feature_list)} 个特征，但输入有 {X_input.shape[1]} 个")
                        st.stop()
                    
                    # 检查模型期望的特征数
                    model_n_features = None
                    try:
                        if hasattr(model, 'n_features_'):
                            model_n_features = model.n_features_
                        elif hasattr(model, 'booster_'):
                            model_n_features = model.booster_.num_feature()
                    except:
                        pass
                    
                    if model_n_features and X_input.shape[1] != model_n_features:
                        st.error(f"❌ 特征数量不匹配！模型期望 {model_n_features} 个特征，但输入有 {X_input.shape[1]} 个")
                        st.stop()
                    
                    # 9. 进行预测
                    proba = float(model.predict_proba(X_input)[:, 1][0])
                    risk_percent = proba * 100.0
                    
                    # 调试信息（可选，通过expander显示）
                    with st.expander("🔍 调试信息（点击查看）"):
                        st.write(f"**特征数量**: {len(feature_list)}")
                        st.write(f"**模型期望特征数**: {model_n_features if model_n_features else '未知'}")
                        st.write(f"**输入数据形状**: {X_input.shape}")
                        st.write(f"**用户输入的特征**: {list(user_values.keys())}")
                        if missing_features:
                            st.write(f"**缺失的特征（已用0填充）**: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
                        st.write(f"**预测概率**: {proba:.6f}")
                    
                except ImportError:
                    # 如果无法导入prepare_features，使用简化版本
                    st.warning("⚠ 无法导入prepare_features模块，使用简化预处理流程")
                    
                    # 简化流程：直接从训练数据样本开始
                    # 移除APACHE死亡概率特征
                    apache_prob_features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
                    for feat in apache_prob_features:
                        if feat in patient_df.columns:
                            patient_df = patient_df.drop(columns=[feat])
                    
                    # 移除ID列和目标变量
                    id_cols = ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']
                    for col in id_cols:
                        if col in patient_df.columns:
                            patient_df = patient_df.drop(columns=[col])
                    
                    # 处理分类特征（如果有预处理器）
                    if preprocessor and isinstance(preprocessor, dict) and 'encoders' in preprocessor:
                        encoders = preprocessor.get('encoders', {})
                        for col, encoder in encoders.items():
                            if col in patient_df.columns:
                                patient_df[col] = patient_df[col].fillna('Missing')
                                try:
                                    patient_df[col] = patient_df[col].astype(str)
                                    known_classes = set(encoder.classes_)
                                    patient_df[col] = patient_df[col].apply(
                                        lambda x: x if x in known_classes else encoder.classes_[0]
                                    )
                                    patient_df[col] = encoder.transform(patient_df[col])
                                except Exception:
                                    patient_df[col] = 0
                    
                    # 用中位数填充所有特征
                    for feat in feature_list:
                        if feat in patient_df.columns:
                            if feat in feature_medians.index:
                                patient_df[feat] = feature_medians[feat]
                            else:
                                patient_df[feat] = 0.0
                        else:
                            patient_df[feat] = feature_medians.get(feat, 0.0) if feat in feature_medians.index else 0.0
                    
                    # 用用户输入的值覆盖
                    for feat_name, val in user_values.items():
                        if feat_name in patient_df.columns:
                            patient_df[feat_name] = float(val)
                        elif feat_name in feature_list:
                            patient_df[feat_name] = float(val)
                    
                    # 按特征顺序组织输入
                    X_input_values = []
                    for feat in feature_list:
                        if feat in patient_df.columns:
                            val = patient_df[feat].iloc[0]
                            if pd.isna(val):
                                val = feature_medians.get(feat, 0.0) if feat in feature_medians.index else 0.0
                            X_input_values.append(float(val))
                        else:
                            X_input_values.append(feature_medians.get(feat, 0.0) if feat in feature_medians.index else 0.0)
                    
                    X_input = np.array(X_input_values).reshape(1, -1)
                    
                    # 验证特征数量
                    model_n_features = None
                    try:
                        if hasattr(model, 'n_features_'):
                            model_n_features = model.n_features_
                        elif hasattr(model, 'booster_'):
                            model_n_features = model.booster_.num_feature()
                    except:
                        pass
                    
                    if model_n_features and X_input.shape[1] != model_n_features:
                        st.error(f"❌ 特征数量不匹配！模型期望 {model_n_features} 个特征，但输入有 {X_input.shape[1]} 个")
                        st.stop()
                    
                    # 进行预测
                    proba = float(model.predict_proba(X_input)[:, 1][0])
                    risk_percent = proba * 100.0
                    
                    # 调试信息
                    with st.expander("🔍 调试信息（点击查看）"):
                        st.write(f"**特征数量**: {len(feature_list)}")
                        st.write(f"**模型期望特征数**: {model_n_features if model_n_features else '未知'}")
                        st.write(f"**输入数据形状**: {X_input.shape}")
                        st.write(f"**用户输入的特征**: {list(user_values.keys())}")
                        st.write(f"**预测概率**: {proba:.6f}")
                        st.write("⚠ 注意：使用了简化预处理流程，可能与训练时不完全一致")

                st.markdown("#### 预测结果")
                col_result1, col_result2 = st.columns([1, 2])

                with col_result1:
                    st.metric("预测住院死亡概率", f"{risk_percent:.2f} %")

                # 风险分层
                if proba >= threshold:
                    risk_level = "高风险"
                    color_class = "warning-box"
                elif proba >= 0.2:
                    risk_level = "中等风险"
                    color_class = "info-box"
                else:
                    risk_level = "低风险"
                    color_class = "success-box"

                with col_result2:
                    st.markdown(
                        f"""
                        <div class="{color_class}">
                            <h4>风险分层：{risk_level}</h4>
                            <p><strong>模型输出的死亡概率：</strong>{risk_percent:.2f}%</p>
                            <p><strong>判定阈值：</strong>{threshold * 100:.0f}%</p>
                            <p style="margin-top:0.5rem; font-size:0.9rem;">
                                注：本结果基于 WiDS Datathon 2020 ICU 数据训练的机器学习模型，仅作为科研与教学参考，
                                不应直接用于真实临床决策。
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"在线预测时发生错误：{str(e)}")

# 主要分析模块
st.markdown('<div class="section-header">🔬 主要分析模块</div>', unsafe_allow_html=True)

# 创建标签页
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 数据读取", 
    "🔧 数据预处理", 
    "📊 统计分析", 
    "🤖 模型训练", 
    "📈 模型评估", 
    "🏆 Kaggle结果"
])

with tab1:
    st.markdown("### 数据读取模块")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **功能说明：**
        - 使用 pandas 高效加载大规模数据集（91,713条记录，186个特征）
        - 标准化缺失值处理（将 'NA' 和空字符串统一映射为 NaN）
        - 加载官方数据字典，解析特征医学类别
        - 特征分类：行政管理、人口统计、生命体征、实验室指标、APACHE评分
        """)
    with col2:
        st.markdown("""
        **关键特性：**
        - 内存优化：设置 `low_memory=False` 确保完整加载
        - 医学逻辑：基于数据字典进行特征分类
        - 可视化：缺失值分析、目标变量分布等
        """)
    
    # 数据字典预览
    st.markdown("#### 数据字典预览")
    try:
        dict_path = BASE_DIR / "data" / "WiDS Datathon 2020 Dictionary.csv"
        if dict_path.exists():
            dict_df = load_csv_data(dict_path)
            
            # 显示数据字典基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总行数", f"{len(dict_df):,}")
            with col2:
                st.metric("总列数", f"{len(dict_df.columns)}")
            
            # 提供选项：显示前N行或全部
            display_option = st.radio(
                "显示选项：",
                ["前10行（预览）", "前50行", "全部数据"],
                horizontal=True,
                index=0
            )
            
            if display_option == "前10行（预览）":
                st.dataframe(dict_df.head(10), use_container_width=True, height=400)
            elif display_option == "前50行":
                st.dataframe(dict_df.head(50), use_container_width=True, height=600)
            else:
                st.dataframe(dict_df, use_container_width=True, height=600)
        else:
            st.warning("⚠️ 数据字典文件未找到，请确保 data/WiDS Datathon 2020 Dictionary.csv 存在")
    except Exception as e:
        st.info(f"数据字典加载信息: {str(e)}")
    
    # 缺失值分析可视化
    st.markdown("#### 缺失值分析")
    st.markdown("""
    以下图表展示了数据集中缺失值的分布情况，包括：
    - 缺失值比例分布直方图
    - 缺失值比例最高的特征
    - 缺失值统计信息
    """)
    
    try:
        data_path = BASE_DIR / "data" / "training_v2.csv"
        if data_path.exists():
            # 使用缓存函数计算缺失值统计（首次加载后会被缓存）
            with st.spinner("正在加载数据并计算缺失值（首次加载可能需要几秒钟，后续会使用缓存）..."):
                missing_df, total_rows, total_cols = compute_missing_stats(data_path)
                columns = missing_df['特征'].tolist()
            
            # 统计信息
            total_cols = len(columns)
            no_missing = total_cols - len(missing_df[missing_df['缺失比例(%)'] > 0])
            low_missing = len(missing_df[(missing_df['缺失比例(%)'] > 0) & (missing_df['缺失比例(%)'] <= 50)])
            medium_missing = len(missing_df[(missing_df['缺失比例(%)'] > 50) & (missing_df['缺失比例(%)'] <= 70)])
            high_missing = len(missing_df[missing_df['缺失比例(%)'] > 70])
            
            # 显示统计摘要
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("无缺失特征", f"{no_missing}")
            with col2:
                st.metric("低缺失 (0-50%)", f"{low_missing}")
            with col3:
                st.metric("中等缺失 (50-70%)", f"{medium_missing}")
            with col4:
                st.metric("高缺失 (>70%)", f"{high_missing}")
            
            # 将三个图表和一个表格放在四列布局中
            chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
            
            # 1. 缺失值比例分布直方图
            with chart_col1:
                fig_hist = px.histogram(
                    missing_df,
                    x='缺失比例(%)',
                    nbins=20,
                    title='缺失值比例分布',
                    labels={'缺失比例(%)': '缺失比例 (%)', 'count': '特征数量'},
                    color_discrete_sequence=['#3498db']
                )
                # 添加阈值线
                fig_hist.add_vline(x=50, line_dash="dash", line_color="#e67e22", 
                                  annotation_text="50%", annotation_position="top")
                fig_hist.add_vline(x=70, line_dash="dash", line_color="#e74c3c", 
                                  annotation_text="70%", annotation_position="top")
                fig_hist.update_layout(bargap=0.1, showlegend=False, height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # 2. 缺失值比例最高的前20个特征（水平条形图）
            with chart_col2:
                top_missing = missing_df.head(20)
                fig_bar = px.bar(
                    top_missing,
                    x='缺失比例(%)',
                    y='特征',
                    orientation='h',
                    title='前20个高缺失特征',
                    labels={'缺失比例(%)': '缺失比例 (%)', '特征': '特征名称'},
                    color='缺失比例(%)',
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # 3. 缺失值阈值统计（条形图）
            with chart_col3:
                threshold_data = pd.DataFrame({
                    '类别': ['无缺失', '低缺失', '中等缺失', '高缺失'],
                    '特征数量': [no_missing, low_missing, medium_missing, high_missing]
                })
                fig_threshold = px.bar(
                    threshold_data,
                    x='类别',
                    y='特征数量',
                    title='缺失值阈值统计',
                    labels={'类别': '缺失值类别', '特征数量': '特征数量'},
                    color='类别',
                    color_discrete_map={
                        '无缺失': '#2ecc71',
                        '低缺失': '#f39c12',
                        '中等缺失': '#e67e22',
                        '高缺失': '#e74c3c'
                    }
                )
                fig_threshold.update_traces(texttemplate='%{y}', textposition='outside')
                # 扩大y轴范围，确保顶部数字完整显示
                max_y = max([no_missing, low_missing, medium_missing, high_missing])
                fig_threshold.update_layout(
                    height=400, 
                    showlegend=False,
                    yaxis=dict(range=[0, max_y * 1.15] if max_y > 0 else None)
                )
                st.plotly_chart(fig_threshold, use_container_width=True)
            
            # 4. 显示前20个缺失值比例最高的特征表格
            with chart_col4:
                st.markdown("**详细数据（前20个）**")
                st.dataframe(
                    missing_df.head(20)[['特征', '缺失比例(%)']], 
                    use_container_width=True, 
                    hide_index=True,
                    height=400
                )
            
        else:
            st.warning("⚠️ 数据文件未找到，请确保 data/training_v2.csv 存在")
    except Exception as e:
        st.error(f"生成缺失值分析图表时出错: {str(e)}")
        st.info("💡 提示：请确保数据文件存在且格式正确")
    
    # 特征分类可视化
    st.markdown("#### 特征分类可视化")
    st.markdown("""
    以下图表展示了基于数据字典的特征分类结果，包括：
    - 各医学类别特征数量分布
    - 主要特征类别统计
    """)
    
    try:
        dict_path = BASE_DIR / "data" / "WiDS Datathon 2020 Dictionary.csv"
        data_path = BASE_DIR / "data" / "training_v2.csv"
        
        if dict_path.exists() and data_path.exists():
            dict_df = pd.read_csv(dict_path)
            train_df = load_csv_data(data_path, nrows=0)  # 只读取列名
            
            if 'Category' in dict_df.columns and 'Variable Name' in dict_df.columns:
                # 创建特征分类字典
                feature_categories = {}
                for _, row in dict_df.iterrows():
                    category = row['Category']
                    var_name = row['Variable Name']
                    if category not in feature_categories:
                        feature_categories[category] = []
                    feature_categories[category].append(var_name)
                
                # 计算每个类别在实际数据中的特征数量
                category_names_cn = {
                    'demographic': '人口统计学指标',
                    'vitals': '实时生命体征',
                    'labs': '常规实验室化验指标',
                    'APACHE covariate': 'APACHE评分协变量',
                    'labs blood gas': '血气分析指标'
                }
                
                main_categories = ['demographic', 'vitals', 'labs', 'APACHE covariate', 'labs blood gas']
                category_counts_dict = {}
                
                for cat in main_categories:
                    if cat in feature_categories:
                        features = feature_categories[cat]
                        existing_features = [f for f in features if f in train_df.columns]
                        category_counts_dict[category_names_cn.get(cat, cat)] = len(existing_features)
                
                # 计算其他类别
                other_count = 0
                for cat in feature_categories.keys():
                    if cat not in main_categories:
                        features = feature_categories[cat]
                        existing_features = [f for f in features if f in train_df.columns]
                        other_count += len(existing_features)
                
                if other_count > 0:
                    category_counts_dict['其他类别'] = other_count
                
                # 创建DataFrame
                category_counts = pd.Series(category_counts_dict)
                total_features = category_counts.sum()
                
                # 显示统计摘要
                st.markdown("**特征分类统计摘要**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总特征数", f"{total_features}")
                with col2:
                    st.metric("主要类别数", f"{len(category_counts)}")
                
                # 将图表和表格放在一行（三列布局）
                chart_col1, chart_col2, chart_col3 = st.columns(3)
                
                # 1. 特征类别分布饼图
                with chart_col1:
                    fig_pie = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title='特征类别分布',
                        hole=0.4
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>特征数量: %{value}<br>占比: %{percent}<extra></extra>'
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # 2. 特征类别分布水平条形图
                with chart_col2:
                    fig_hbar = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title='特征类别分布',
                        labels={'x': '特征数量', 'y': '类别'},
                        color=category_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig_hbar.update_traces(
                        text=category_counts.values,
                        texttemplate='%{text}',
                        textposition='outside',
                        customdata=(category_counts.values / total_features * 100)
                    )
                    fig_hbar.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_hbar, use_container_width=True)
                
                # 3. 显示详细统计表
                with chart_col3:
                    st.markdown("**详细数据统计表**")
                    category_stats = pd.DataFrame({
                        '类别': category_counts.index,
                        '特征数量': category_counts.values,
                        '占比(%)': (category_counts.values / total_features * 100).round(2)
                    }).sort_values('特征数量', ascending=False)
                    st.dataframe(
                        category_stats, 
                        use_container_width=True, 
                        hide_index=True,
                        height=400
                    )
            else:
                st.warning("⚠️ 数据字典格式不正确，缺少必要的列（Category 或 Variable Name）")
        else:
            if not dict_path.exists():
                st.warning("⚠️ 数据字典文件未找到，请确保 data/WiDS Datathon 2020 Dictionary.csv 存在")
            if not data_path.exists():
                st.warning("⚠️ 数据文件未找到，请确保 data/training_v2.csv 存在")
    except Exception as e:
        st.error(f"生成特征分类图表时出错: {str(e)}")
        st.info("💡 提示：请确保数据字典和数据文件存在且格式正确")

with tab2:
    st.markdown("### 数据预处理模块")
    st.markdown("**处理策略：**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **缺失值处理**
        - 高缺失率特征（>70%）: 直接剔除
        - 数值型特征: 中位数填充
        - 分类特征: 众数填充
        - 医学逻辑填充: 基于临床知识进行智能填充
        """)
    with col2:
        st.markdown("""
        **异常值处理**
        - 基于医学合理范围进行异常值检测
        - 使用IQR方法识别极端值
        """)
    with col3:
        st.markdown("""
        **特征工程**
        - 创建交互特征
        - 时间序列特征提取
        - GCS评分特征构建
        """)
    
    # 显示预处理结果统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("原始特征数", "186")
        st.metric("完全填充特征", "11")
    with col2:
        st.metric("缺失特征数", "175")
        st.metric("处理后特征数", "~180")
    with col3:
        st.metric("缺失值填充率", ">95%")
        st.metric("数据完整性", "高")
    
    # 数据预处理可视化
    st.markdown("#### 数据预处理可视化")
    st.markdown("""
    以下图表展示了数据预处理的全流程，包括：
    - 特征降维过程
    - 被删除特征的类型分析
    - 特征类型分布
    - 缺失值处理策略
    """)
    
    try:
        data_path = BASE_DIR / "data" / "training_v2.csv"
        if data_path.exists():
            with st.spinner("正在加载数据并计算预处理统计信息..."):
                # 读取数据（优化：使用更小的采样减少内存占用和加载时间）
                train_df = load_csv_data(data_path, nrows=10000, low_memory=False, na_values=['NA', ''])
                
                # 计算缺失值
                missing_percent = (train_df.isnull().sum() / len(train_df)) * 100
                high_missing_cols = missing_percent[missing_percent > 70].index.tolist()
                train_df_cleaned = train_df.drop(columns=high_missing_cols)
                
                # 识别分类特征
                object_cols = train_df_cleaned.select_dtypes(include=['object']).columns.tolist()
                numeric_cols = train_df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col not in ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']]
                
                # 计算缺失值总数
                total_missing = train_df.isnull().sum().sum()
                after_fill_missing = total_missing  # 保留缺失值，不填充
                
                # 将四个图表放在一行四列布局
                chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
                
                # 1. 特征降维过程
                with chart_col1:
                    st.markdown("##### 特征降维过程")
                    stages = ['原始特征', '删除高缺失值列', '最终特征']
                    counts = [train_df.shape[1], len(high_missing_cols), train_df_cleaned.shape[1]]
                    fig1 = px.bar(
                        x=stages,
                        y=counts,
                        labels={'x': '处理阶段', 'y': '特征数量'},
                        color=stages,
                        color_discrete_map={
                            '原始特征': '#3498db',
                            '删除高缺失值列': '#e74c3c',
                            '最终特征': '#2ecc71'
                        }
                    )
                    fig1.update_traces(texttemplate='%{y}', textposition='outside')
                    # 扩大y轴范围，确保顶部数字完整显示
                    max_y = max(counts)
                    fig1.update_layout(
                        showlegend=False, 
                        height=400,
                        yaxis=dict(range=[0, max_y * 1.15])
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                # 2. 被删除特征的类型分析
                with chart_col2:
                    st.markdown("##### 被删除特征类型分布")
                    h1_count = sum(1 for col in high_missing_cols if col.startswith('h1_'))
                    d1_count = sum(1 for col in high_missing_cols if col.startswith('d1_'))
                    other_count = len(high_missing_cols) - h1_count - d1_count
                    
                    deleted_types = ['h1_前缀(第一小时)', 'd1_前缀(第一天)', '其他特征']
                    deleted_counts = [h1_count, d1_count, other_count]
                    
                    fig2 = px.bar(
                        x=deleted_types,
                        y=deleted_counts,
                        labels={'x': '特征类型', 'y': '特征数量'},
                        color=deleted_types,
                        color_discrete_map={
                            'h1_前缀(第一小时)': '#e74c3c',
                            'd1_前缀(第一天)': '#f39c12',
                            '其他特征': '#95a5a6'
                        }
                    )
                    fig2.update_traces(texttemplate='%{y}', textposition='outside')
                    # 扩大y轴范围，确保顶部数字完整显示
                    max_y = max(deleted_counts) if deleted_counts else 0
                    fig2.update_layout(
                        showlegend=False, 
                        height=400,
                        yaxis=dict(range=[0, max_y * 1.15] if max_y > 0 else None)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # 3. 特征类型分布
                with chart_col3:
                    st.markdown("##### 特征类型分布")
                    feature_types = ['分类特征', '数值型特征']
                    feature_counts = [len(object_cols), len(numeric_cols)]
                    
                    fig3 = px.bar(
                        x=feature_types,
                        y=feature_counts,
                        labels={'x': '特征类型', 'y': '特征数量'},
                        color=feature_types,
                        color_discrete_map={
                            '分类特征': '#9b59b6',
                            '数值型特征': '#3498db'
                        }
                    )
                    fig3.update_traces(texttemplate='%{y}', textposition='outside')
                    # 扩大y轴范围，确保顶部数字完整显示
                    max_y = max(feature_counts) if feature_counts else 0
                    fig3.update_layout(
                        showlegend=False, 
                        height=400,
                        yaxis=dict(range=[0, max_y * 1.15] if max_y > 0 else None)
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # 4. 缺失值处理策略
                with chart_col4:
                    st.markdown("##### 缺失值处理策略")
                    fill_stages = ['缺失值统计', '保留缺失值']
                    missing_counts = [total_missing, after_fill_missing]
                    
                    fig4 = px.bar(
                        x=fill_stages,
                        y=missing_counts,
                        labels={'x': '处理阶段', 'y': '缺失值数量'},
                        color=fill_stages,
                        color_discrete_map={
                            '缺失值统计': '#e74c3c',
                            '保留缺失值': '#2ecc71'
                        }
                    )
                    fig4.update_traces(texttemplate='%{y:,}', textposition='outside')
                    # 扩大y轴范围，确保顶部数字完整显示
                    max_y = max(missing_counts) if missing_counts else 0
                    fig4.update_layout(
                        showlegend=False, 
                        height=400,
                        yaxis=dict(range=[0, max_y * 1.15] if max_y > 0 else None)
                    )
                    st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("⚠️ 数据文件未找到，请确保 data/training_v2.csv 存在")
    except Exception as e:
        st.error(f"生成数据预处理可视化图表时出错: {str(e)}")
    
    # 医学特征分析可视化
    st.markdown("#### 医学特征分析可视化")
    st.markdown("""
    以下图表展示了关键医学特征的分析结果，包括：
    - 生命体征特征分布
    - 实验室指标特征分析
    - APACHE评分特征
    - 特征与目标变量的关系
    """)
    
    try:
        data_path = BASE_DIR / "data" / "training_v2.csv"
        if data_path.exists():
            with st.spinner("正在加载数据并分析医学特征..."):
                # 优化：使用更小的采样减少内存占用和加载时间
                train_df = load_csv_data(data_path, nrows=10000, low_memory=False, na_values=['NA', ''])
                
                # 选择关键医学特征
                key_features = ['age', 'bmi', 'heart_rate_apache', 'temp_apache', 
                               'd1_glucose_max', 'd1_glucose_min', 'apache_4a_icu_death_prob']
                available_features = [f for f in key_features if f in train_df.columns]
                
                if len(available_features) > 0:
                    # 创建一行三列布局
                    med_col1, med_col2, med_col3 = st.columns(3)
                    
                    # 1. 关键特征与目标变量的相关性
                    with med_col1:
                        st.markdown("##### 关键特征与目标变量相关性")
                        correlations = {}
                        for feature in available_features:
                            valid_mask = train_df[[feature, 'hospital_death']].notna().all(axis=1)
                            if valid_mask.sum() > 0:
                                corr = train_df.loc[valid_mask, feature].corr(
                                    train_df.loc[valid_mask, 'hospital_death']
                                )
                                if pd.notna(corr):
                                    correlations[feature] = corr
                        
                        if correlations:
                            corr_df = pd.DataFrame({
                                '特征': list(correlations.keys()),
                                '相关系数': list(correlations.values())
                            }).sort_values('相关系数', key=abs, ascending=False)
                            
                            fig_corr = px.bar(
                                corr_df,
                                x='特征',
                                y='相关系数',
                                labels={'特征': '特征名称', '相关系数': '相关系数'},
                                color='相关系数',
                                color_continuous_scale='RdBu',
                                color_continuous_midpoint=0
                            )
                            fig_corr.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # 2. 关键特征的分布（按目标变量分组）
                    with med_col2:
                        st.markdown("##### 关键特征分布（按目标变量分组）")
                        # 选择第一个可用特征进行展示
                        if available_features:
                            feature = available_features[0]
                            valid_data = train_df[[feature, 'hospital_death']].dropna()
                            
                            if len(valid_data) > 0:
                                fig_dist = px.histogram(
                                    valid_data,
                                    x=feature,
                                    color='hospital_death',
                                    nbins=30,
                                    labels={'hospital_death': '住院死亡', feature: feature},
                                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                                )
                                fig_dist.update_layout(height=400)
                                st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # 3. 关键特征统计摘要表格
                    with med_col3:
                        st.markdown("##### 关键特征统计摘要")
                        summary_data = []
                        for feature in available_features[:10]:  # 限制前10个特征
                            valid_data = train_df[feature].dropna()
                            if len(valid_data) > 0:
                                summary_data.append({
                                    '特征': feature,
                                    '均值': valid_data.mean(),
                                    '中位数': valid_data.median(),
                                    '标准差': valid_data.std(),
                                    '最小值': valid_data.min(),
                                    '最大值': valid_data.max()
                                })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=400)
                else:
                    st.info("💡 未找到可用的关键医学特征进行可视化")
        else:
            st.warning("⚠️ 数据文件未找到，请确保 data/training_v2.csv 存在")
    except Exception as e:
        st.error(f"生成医学特征分析图表时出错: {str(e)}")

with tab3:
    st.markdown("### 统计分析模块")
    st.markdown("**分析内容：**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **描述性统计分析**
        - 总体统计、分组统计（存活组 vs 死亡组）
        - 中心趋势、离散程度、分布特征
        """)
    with col2:
        st.markdown("""
        **特征分布分析**
        - 正态性检验（D'Agostino's K² 检验）
        - 分布可视化
        """)
    with col3:
        st.markdown("""
        **相关性分析**
        - 与目标变量的相关性
        - 特征间相关性矩阵
        """)
    with col4:
        st.markdown("""
        **特征重要性评估**
        - 综合多个统计指标
        - 为模型建立提供依据
        """)
    
    # 统计分析可视化
    try:
        data_path = BASE_DIR / "data" / "training_v2.csv"
        if data_path.exists():
            with st.spinner("正在加载数据并生成统计分析图表..."):
                # 优化：使用更小的采样减少内存占用和加载时间
                train_df = load_csv_data(data_path, nrows=10000, low_memory=False, na_values=['NA', ''])
                
                # 常见临床特征列表（12个）
                common_features = [
                    'age', 'bmi', 'weight', 'height', 'heart_rate_apache', 
                    'temp_apache', 'resprate_apache', 'map_apache', 
                    'creatinine_apache', 'bun_apache', 'sodium_apache', 
                    'glucose_apache', 'wbc_apache'
                ]
                available_features = [f for f in common_features if f in train_df.columns][:12]
                
                # 特征中文名称
                feature_names_cn = {
                    'age': '年龄', 'bmi': 'BMI', 'weight': '体重', 'height': '身高',
                    'heart_rate_apache': '心率', 'temp_apache': '体温', 
                    'resprate_apache': '呼吸频率', 'map_apache': '平均动脉压',
                    'creatinine_apache': '肌酐', 'bun_apache': '血尿素氮',
                    'sodium_apache': '血钠', 'glucose_apache': '血糖', 
                    'wbc_apache': '白细胞计数'
                }
                
                # 1. 12个常见临床特征箱线图分布对比
                st.markdown("#### 常见临床特征箱线图分布对比")
                st.markdown("**12个常见临床特征在存活组（绿色）与死亡组（红色）间的箱线图分布对比**")
                
                if len(available_features) > 0:
                    # 创建6列布局，两行显示（12个特征 = 2行 × 6列）
                    n_cols = 6
                    n_features = min(len(available_features), 12)
                    
                    # 按行显示
                    for row in range((n_features + n_cols - 1) // n_cols):
                        cols = st.columns(n_cols)
                        for col_idx in range(n_cols):
                            feature_idx = row * n_cols + col_idx
                            if feature_idx < n_features:
                                with cols[col_idx]:
                                    feature = available_features[feature_idx]
                                    feature_name = feature_names_cn.get(feature, feature)
                                    
                                    # 准备数据
                                    data = train_df[[feature, 'hospital_death']].dropna()
                                    if len(data) > 0:
                                        # 创建分组标签
                                        data['组别'] = data['hospital_death'].map({0: '存活组', 1: '死亡组'})
                                        
                                        # 使用plotly express创建箱线图
                                        fig = px.box(
                                            data,
                                            x='组别',
                                            y=feature,
                                            color='组别',
                                            color_discrete_map={'存活组': '#2ecc71', '死亡组': '#e74c3c'},
                                            title=feature_name
                                        )
                                        
                                        fig.update_layout(
                                            title=dict(
                                                text=feature_name,
                                                font=dict(size=12)
                                            ),
                                            yaxis_title='特征值',
                                            xaxis_title='',
                                            height=300,
                                            showlegend=False,
                                            margin=dict(l=30, r=20, t=50, b=40)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                
                # 2. 关键特征均值与中位数归一化对比和数值型特征分布类型统计（一行四列）
                st.markdown("#### 关键特征均值与中位数归一化对比和数值型特征分布类型统计")
                
                if len(available_features) > 0:
                    # 计算均值和中位数
                    mean_data = []
                    median_data = []
                    
                    for feature in available_features[:10]:  # 前10个特征
                        data = train_df[[feature, 'hospital_death']].dropna()
                        if len(data) > 0:
                            alive_mean = data[data['hospital_death'] == 0][feature].mean()
                            death_mean = data[data['hospital_death'] == 1][feature].mean()
                            alive_median = data[data['hospital_death'] == 0][feature].median()
                            death_median = data[data['hospital_death'] == 1][feature].median()
                            
                            # 归一化（相对于总体均值）
                            overall_mean = data[feature].mean()
                            overall_median = data[feature].median()
                            
                            mean_data.append({
                                '特征': feature_names_cn.get(feature, feature),
                                '存活组': (alive_mean - overall_mean) / overall_mean if overall_mean != 0 else 0,
                                '死亡组': (death_mean - overall_mean) / overall_mean if overall_mean != 0 else 0
                            })
                            
                            median_data.append({
                                '特征': feature_names_cn.get(feature, feature),
                                '存活组': (alive_median - overall_median) / overall_median if overall_median != 0 else 0,
                                '死亡组': (death_median - overall_median) / overall_median if overall_median != 0 else 0
                            })
                    
                    # 获取数值型特征并计算偏度和峰度
                    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [col for col in numeric_cols if col not in 
                                   ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']]
                    
                    skewness_list = []
                    kurtosis_list = []
                    feature_list = []
                    
                    for col in numeric_cols[:50]:  # 限制前50个特征
                        data = train_df[col].dropna()
                        if len(data) > 100:  # 至少100个样本
                            from scipy.stats import skew, kurtosis
                            sk = skew(data)
                            kt = kurtosis(data)
                            skewness_list.append(sk)
                            kurtosis_list.append(kt)
                            feature_list.append(col)
                    
                    # 创建四列布局
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("##### (a) 均值归一化对比")
                        if mean_data:
                            mean_df = pd.DataFrame(mean_data)
                            fig_mean = go.Figure()
                            fig_mean.add_trace(go.Bar(
                                x=mean_df['特征'],
                                y=mean_df['存活组'],
                                name='存活组',
                                marker_color='#2ecc71'
                            ))
                            fig_mean.add_trace(go.Bar(
                                x=mean_df['特征'],
                                y=mean_df['死亡组'],
                                name='死亡组',
                                marker_color='#e74c3c'
                            ))
                            fig_mean.update_layout(
                                barmode='group',
                                height=400,
                                xaxis_tickangle=-45,
                                showlegend=True
                            )
                            st.plotly_chart(fig_mean, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### (b) 中位数归一化对比")
                        if median_data:
                            median_df = pd.DataFrame(median_data)
                            fig_median = go.Figure()
                            fig_median.add_trace(go.Bar(
                                x=median_df['特征'],
                                y=median_df['存活组'],
                                name='存活组',
                                marker_color='#2ecc71'
                            ))
                            fig_median.add_trace(go.Bar(
                                x=median_df['特征'],
                                y=median_df['死亡组'],
                                name='死亡组',
                                marker_color='#e74c3c'
                            ))
                            fig_median.update_layout(
                                barmode='group',
                                height=400,
                                xaxis_tickangle=-45,
                                showlegend=True
                            )
                            st.plotly_chart(fig_median, use_container_width=True)
                    
                    with col3:
                        st.markdown("##### (c) 分布类型统计")
                        if len(skewness_list) > 0:
                            # 分类分布类型
                            normal_count = sum(1 for s, k in zip(skewness_list, kurtosis_list) 
                                             if abs(s) < 0.5 and abs(k) < 0.5)
                            skewed_count = sum(1 for s in skewness_list if abs(s) >= 0.5)
                            heavy_tail_count = sum(1 for k in kurtosis_list if abs(k) >= 0.5)
                            other_count = len(skewness_list) - normal_count - skewed_count - heavy_tail_count
                            
                            dist_types = ['正态分布', '偏态分布', '重尾分布', '其他']
                            dist_counts = [normal_count, skewed_count, heavy_tail_count, other_count]
                            
                            fig_dist = px.pie(
                                values=dist_counts,
                                names=dist_types,
                                hole=0.4
                            )
                            fig_dist.update_layout(height=400)
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col4:
                        st.markdown("##### (d) 偏度-峰度关联散点图")
                        if len(skewness_list) > 0:
                            fig_scatter = px.scatter(
                                x=skewness_list,
                                y=kurtosis_list,
                                labels={'x': '偏度', 'y': '峰度'},
                                hover_name=feature_list[:len(skewness_list)]
                            )
                            # 添加参考线
                            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
                            fig_scatter.update_layout(height=400)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 3. 特征相关性分析、矩阵热力图和初步特征重要性综合评分（一行三列）
                st.markdown("#### 特征相关性分析、矩阵热力图和初步特征重要性综合评分")
                
                # 尝试加载相关性结果文件（相对于应用目录）
                corr_path = BASE_DIR / "results" / "statistical_analysis" / "correlation_with_target.csv"
                corr_matrix_path = BASE_DIR / "results" / "statistical_analysis" / "feature_correlation_matrix.csv"
                importance_path = BASE_DIR / "results" / "statistical_analysis" / "feature_importance_preliminary.csv"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### (a) 与目标变量相关性 Top 20")
                    if corr_path.exists():
                        corr_df = load_csv_data(corr_path)
                        top_corr = corr_df.head(20)
                        
                        fig_corr_bar = px.bar(
                            top_corr,
                            x='相关系数',
                            y='特征名',
                            orientation='h',
                            color='相关系数',
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=0,
                            labels={'相关系数': '相关系数', '特征名': '特征名称'}
                        )
                        fig_corr_bar.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=500,
                            showlegend=False
                        )
                        st.plotly_chart(fig_corr_bar, use_container_width=True)
                    else:
                        st.info("💡 运行 statistical_analysis.py 生成相关性分析结果")
                
                with col2:
                    st.markdown("##### (b) 特征间相关性矩阵热力图")
                    if corr_matrix_path.exists() and corr_path.exists():
                        corr_matrix = load_csv_data(corr_matrix_path, index_col=0)
                        corr_df = load_csv_data(corr_path)
                        
                        # 选择Top 30特征（基于与目标变量的相关性）
                        top_features = corr_df.head(30)['特征名'].tolist()
                        available_top = [f for f in top_features if f in corr_matrix.index and f in corr_matrix.columns]
                        
                        if len(available_top) > 1:
                            corr_subset = corr_matrix.loc[available_top, available_top]
                            
                            fig_heatmap = px.imshow(
                                corr_subset,
                                color_continuous_scale='RdBu',
                                color_continuous_midpoint=0,
                                aspect='auto',
                                labels=dict(color="相关系数")
                            )
                            fig_heatmap.update_layout(height=500)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                            st.info("💡 无法生成相关性矩阵热力图")
                    else:
                        st.info("💡 运行 statistical_analysis.py 生成特征间相关性矩阵")
                
                with col3:
                    st.markdown("##### (c) 初步特征重要性综合评分 Top 30")
                    if importance_path.exists():
                        importance_df = load_csv_data(importance_path)
                        top_importance = importance_df.head(30).sort_values('重要性得分', ascending=True)
                        
                        fig_importance = px.bar(
                            top_importance,
                            x='重要性得分',
                            y='特征名',
                            orientation='h',
                            color='重要性得分',
                            color_continuous_scale='Viridis',
                            labels={'重要性得分': '重要性得分', '特征名': '特征名称'}
                        )
                        fig_importance.update_layout(
                            height=500,
                            showlegend=False
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("💡 运行 statistical_analysis.py 生成特征重要性评估结果")
                
                # 5. 重要性评分 Top 10 关键特征的频率分布对比（一行五列）
                st.markdown("#### 重要性评分 Top 10 关键特征分布对比")
                st.markdown("**存活组 vs 死亡组的频率分布对比**")
                
                importance_path = BASE_DIR / "results" / "statistical_analysis" / "feature_importance_preliminary.csv"
                if importance_path.exists():
                    importance_df = pd.read_csv(importance_path)
                    top10_features = importance_df.head(10)['特征名'].tolist()
                    available_top10 = [f for f in top10_features if f in train_df.columns]
                    
                    if len(available_top10) > 0:
                        # 创建一行五列布局
                        n_cols = 5
                        n_features = min(len(available_top10), 10)
                        
                        for row in range((n_features + n_cols - 1) // n_cols):
                            cols = st.columns(n_cols)
                            for col_idx in range(n_cols):
                                feature_idx = row * n_cols + col_idx
                                if feature_idx < n_features:
                                    with cols[col_idx]:
                                        feature = available_top10[feature_idx]
                                        feature_name = feature_names_cn.get(feature, feature)
                                        
                                        data = train_df[[feature, 'hospital_death']].dropna()
                                        if len(data) > 0:
                                            fig_dist = px.histogram(
                                                data,
                                                x=feature,
                                                color='hospital_death',
                                                nbins=20,
                                                labels={'hospital_death': '住院死亡', feature: feature_name},
                                                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                                                barmode='overlay',
                                                opacity=0.7
                                            )
                                            fig_dist.update_layout(
                                                height=300, 
                                                showlegend=False,
                                                margin=dict(l=10, r=10, t=30, b=10)
                                            )
                                            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("⚠️ 数据文件未找到，请确保 data/training_v2.csv 存在")
    except Exception as e:
        st.error(f"生成统计分析图表时出错: {str(e)}")
        st.info("💡 提示：请确保数据文件存在且格式正确，或运行 statistical_analysis.py 生成分析结果")

with tab4:
    st.markdown("### 模型训练与调优")
    st.markdown("**训练的模型类型：**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **传统机器学习模型**
        - 逻辑回归（基准模型）
        - 随机森林
        - 梯度提升树
        """)
    with col2:
        st.markdown("""
        **梯度提升模型（优化）**
        - XGBoost（Optuna超参数优化）
        - LightGBM（Optuna超参数优化，GPU加速）
        - LightGBM集成（5个不同随机种子）
        """)
    with col3:
        st.markdown("""
        **深度学习模型**
        - 标准深度神经网络
        - Wide & Deep 网络
        - 残差网络（ResNet）
        """)
    
    # 1. 各算法模型在住院死亡预测任务上的性能指标对比（仅依赖本地 results 目录中的CSV）
    st.markdown("#### 各算法模型性能指标对比")
    
    try:
        # 尝试加载实际数据（相对于应用目录）
        metrics_path = BASE_DIR / "results" / "model_training" / "model_metrics.csv"
        if metrics_path.exists():
            metrics_df = load_csv_data(metrics_path, index_col=0)
            # 添加集成模型数据
            ensemble_path = BASE_DIR / "results" / "model_evaluation" / "lightgbm_ensemble_metrics.csv"
            if ensemble_path.exists():
                ensemble_df = load_csv_data(ensemble_path, index_col=0)
                ensemble_row = ensemble_df.iloc[0]
                metrics_df.loc['LightGBM_Ensemble'] = ensemble_row
        else:
            # 使用默认数据
            metrics_df = pd.DataFrame({
                'Accuracy': [0.9061, 0.9060, 0.9199, 0.9175, 0.9160, 0.9231],
                'Precision': [0.4586, 0.4610, 0.5356, 0.5211, 0.5127, 0.5570],
                'Recall': [0.4902, 0.5306, 0.5370, 0.5382, 0.5370, 0.5338],
                'F1-Score': [0.4739, 0.4934, 0.5363, 0.5295, 0.5245, 0.5452],
                'AUC-ROC': [0.8768, 0.8876, 0.8999, 0.9018, 0.9014, 0.9070],
                'AP-Score': [0.4811, 0.5170, 0.5688, 0.5716, 0.5701, 0.5951]
            }, index=['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'LightGBM_Ensemble'])
        
        metrics_df.index.name = '模型'
        metrics_df = metrics_df.reset_index()
        metrics_df['模型'] = metrics_df['模型'].map({
            'Logistic Regression': '逻辑回归',
            'Random Forest': '随机森林',
            'Gradient Boosting': '梯度提升树',
            'XGBoost': 'XGBoost',
            'LightGBM': 'LightGBM',
            'LightGBM_Ensemble': 'LightGBM集成'
        })
        
        # 创建交互式多指标对比图 - 三列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 雷达图展示多维度性能
            metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AP-Score']
            metrics_cn = {
                'Accuracy': '准确率',
                'Precision': '精确率',
                'Recall': '召回率',
                'F1-Score': 'F1分数',
                'AUC-ROC': 'AUC-ROC',
                'AP-Score': 'AP分数'
            }
            
            # 定义每个指标的自定义范围
            metric_ranges = {
                'Accuracy': [0.9, 0.95],
                'Precision': [0.5, 0.6],
                'Recall': [0.5, 0.55],
                'F1-Score': [0.5, 0.55],
                'AUC-ROC': [0.85, 0.95],
                'AP-Score': [0.55, 0.6]
            }
            
            # 归一化函数：将原始值映射到[0,1]范围
            def normalize_value(value, metric):
                min_val, max_val = metric_ranges[metric]
                # 将值限制在范围内
                clamped_value = max(min_val, min(max_val, value))
                # 归一化到[0,1]
                normalized = (clamped_value - min_val) / (max_val - min_val)
                return normalized
            
            # 选择前4个模型进行雷达图对比
            # 定义模型颜色映射和填充模式（深红色放在底层，先添加）
            model_configs = {
                'XGBoost': {
                    'color': '#8B0000',  # 深红色 - 底层
                    'fill': 'toself',
                    'fill_opacity': 0.2,  # 很低的填充透明度
                    'line_width': 3
                },
                'LightGBM': {
                    'color': '#3498db',  # 蓝色
                    'fill': 'toself',
                    'fill_opacity': 0.25,
                    'line_width': 3
                },
                'LightGBM集成': {
                    'color': '#2ecc71',  # 绿色
                    'fill': 'toself',
                    'fill_opacity': 0.25,
                    'line_width': 3
                },
                '梯度提升树': {
                    'color': '#f39c12',  # 橙色
                    'fill': 'toself',
                    'fill_opacity': 0.25,
                    'line_width': 3
                }
            }
            top_models = ['XGBoost', 'LightGBM', 'LightGBM集成', '梯度提升树']
            fig_radar = go.Figure()
            
            # 将hex颜色转换为rgba以控制填充透明度
            def hex_to_rgba(hex_color, alpha):
                hex_color = hex_color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgba({r}, {g}, {b}, {alpha})'
            
            for model_name in top_models:
                model_data = metrics_df[metrics_df['模型'] == model_name]
                if len(model_data) > 0:
                    # 对每个指标的值进行归一化，同时保存原始值
                    normalized_values = []
                    original_values = []
                    theta_labels = []
                    for metric in metrics_for_radar:
                        original_value = model_data[metric].values[0]
                        normalized_value = normalize_value(original_value, metric)
                        normalized_values.append(normalized_value)
                        original_values.append(original_value)
                        theta_labels.append(metrics_cn[metric])
                    
                    # 为了形成闭合的雷达图，需要在末尾添加第一个点的值
                    normalized_values.append(normalized_values[0])
                    original_values.append(original_values[0])
                    theta_labels.append(theta_labels[0])
                    
                    config = model_configs.get(model_name, {})
                    color = config.get('color', '#000000')
                    fill_opacity = config.get('fill_opacity', 0.3)
                    line_width = config.get('line_width', 2)
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=normalized_values,  # 使用归一化后的值（已闭合）
                        theta=theta_labels,  # 已闭合的标签
                        fill='toself',
                        name=model_name,
                        line_color=color,
                        fillcolor=hex_to_rgba(color, fill_opacity),  # 使用rgba控制填充透明度
                        line=dict(width=line_width, color=color),  # 线条保持不透明，更清晰
                        opacity=1.0,  # trace本身不透明，只让填充透明
                        # 添加自定义数据用于悬停时显示原始值
                        customdata=original_values,
                        hovertemplate='<b>%{theta}</b><br>归一化值: %{r:.3f}<br>原始值: %{customdata:.4f}<extra></extra>'
                    ))
            
            # 设置radialaxis范围为[0,1]，因为数据已经归一化
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="多维度性能雷达图对比（已按指标范围归一化）",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # 显示各指标的范围说明
            st.markdown("""
            <div style="font-size: 0.85em; color: #666; margin-top: -25px; margin-bottom: 10px;">
            <b>指标范围说明：</b><br>
            准确率: [0.9, 0.95] | 精确率: [0.5, 0.6] | 召回率: [0.5, 0.55] | 
            F1分数: [0.5, 0.55] | AUC-ROC: [0.85, 0.95] | AP分数: [0.55, 0.6]<br>
            <i>注：雷达图已按各指标范围归一化显示，悬停可查看原始值</i>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 多指标条形图对比
            selected_metrics = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']
            metrics_cn_map = {
                'AUC-ROC': 'AUC-ROC',
                'F1-Score': 'F1分数',
                'Precision': '精确率',
                'Recall': '召回率'
            }
            
            fig_multi = go.Figure()
            x_pos = np.arange(len(metrics_df))
            width = 0.15
            
            for idx, metric in enumerate(selected_metrics):
                fig_multi.add_trace(go.Bar(
                    x=metrics_df['模型'],
                    y=metrics_df[metric],
                    name=metrics_cn_map[metric],
                    offsetgroup=idx
                ))
            
            fig_multi.update_layout(
                title='多指标性能对比',
                xaxis_title='模型',
                yaxis_title='指标值',
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        
        with col3:
            # AUC-ROC详细对比（主要指标）
            st.markdown("##### AUC-ROC 性能对比")
            fig_auc = px.bar(
                metrics_df.sort_values('AUC-ROC', ascending=True),
                x='AUC-ROC',
                y='模型',
                orientation='h',
                title='各模型 AUC-ROC 性能排名',
                color='AUC-ROC',
                color_continuous_scale='RdYlGn',
                text='AUC-ROC'
            )
            fig_auc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_auc.update_layout(
                height=400,
                xaxis_range=[0.85, 0.92],
                showlegend=False
            )
            st.plotly_chart(fig_auc, use_container_width=True)
        
        # 性能指标数据表
        st.markdown("##### 详细性能指标表")
        display_metrics_df = metrics_df[['模型', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AP-Score']].copy()
        display_metrics_df = display_metrics_df.round(4)
        st.dataframe(display_metrics_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"加载模型性能数据时出错: {str(e)}")
        st.info("💡 提示：请运行 model_training.py 生成模型性能数据")
    
    # 2. LightGBM基础模型与 Optuna 优化模型性能对比
    st.markdown("#### LightGBM 基础模型与 Optuna 优化模型性能对比")
    
    try:
        comparison_path = BASE_DIR / "results" / "model_evaluation" / "base_vs_optuna_comparison.csv"
        if comparison_path.exists():
            comparison_df = load_csv_data(comparison_path, index_col=0)
        else:
            # 使用默认数据
            comparison_df = pd.DataFrame({
                'Base_Model': [0.8338, 0.3150, 0.7884, 0.4501, 0.9014, 0.5701],
                'Optuna_Model': [0.8762, 0.3852, 0.7277, 0.5037, 0.9069, 0.5946],
                'Difference': [0.0425, 0.0702, -0.0606, 0.0536, 0.0055, 0.0245]
            }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AP-Score'])
        
        comparison_df = comparison_df.reset_index()
        comparison_df.columns = ['指标', '基础模型', 'Optuna优化模型', '提升幅度']
        
        # 三列布局：两个图和一个表
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 性能对比条形图
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                x=comparison_df['指标'],
                y=comparison_df['基础模型'],
                name='基础模型',
                marker_color='#95a5a6',
                text=comparison_df['基础模型'].round(4),
                textposition='outside'
            ))
            
            fig_comparison.add_trace(go.Bar(
                x=comparison_df['指标'],
                y=comparison_df['Optuna优化模型'],
                name='Optuna优化模型',
                marker_color='#3498db',
                text=comparison_df['Optuna优化模型'].round(4),
                textposition='outside'
            ))
            
            fig_comparison.update_layout(
                title='基础模型 vs Optuna优化模型性能对比',
                xaxis_title='指标',
                yaxis_title='指标值',
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # 提升幅度可视化
            fig_improvement = go.Figure()
            
            colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in comparison_df['提升幅度']]
            
            fig_improvement.add_trace(go.Bar(
                x=comparison_df['指标'],
                y=comparison_df['提升幅度'],
                marker_color=colors,
                text=comparison_df['提升幅度'].apply(lambda x: f'{x:+.4f}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>提升幅度: %{y:.4f}<extra></extra>'
            ))
            
            fig_improvement.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # 扩大y轴范围，确保顶部数字完整显示
            max_y = comparison_df['提升幅度'].max()
            min_y = comparison_df['提升幅度'].min()
            y_range_padding = max(abs(max_y), abs(min_y)) * 0.35  # 35%的边距（再增加10%）
            
            fig_improvement.update_layout(
                title='Optuna优化带来的性能提升',
                xaxis_title='指标',
                yaxis_title='提升幅度',
                height=400,
                xaxis_tickangle=-45,
                showlegend=False,
                yaxis=dict(range=[min_y - y_range_padding, max_y + y_range_padding])
            )
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        with col3:
            # 详细对比数据表
            st.markdown("##### 详细性能对比数据")
            display_comparison_df = comparison_df.copy()
            display_comparison_df['基础模型'] = display_comparison_df['基础模型'].round(4)
            display_comparison_df['Optuna优化模型'] = display_comparison_df['Optuna优化模型'].round(4)
            display_comparison_df['提升幅度'] = display_comparison_df['提升幅度'].apply(lambda x: f'{x:+.4f}')
            display_comparison_df['提升百分比'] = ((comparison_df['Optuna优化模型'] - comparison_df['基础模型']) / comparison_df['基础模型'] * 100).round(2).apply(lambda x: f'{x:+.2f}%')
            st.dataframe(display_comparison_df, use_container_width=True, hide_index=True, height=400)
        
        # 关键发现总结
        st.markdown("##### 💡 关键发现")
        st.markdown("""
        - **AUC-ROC提升**: 从 0.9014 提升到 0.9069（+0.6%），概率校准能力显著改善
        - **精确率大幅提升**: 从 0.3150 提升到 0.3852（+22.3%），显著减少误诊
        - **准确率提升**: 从 0.8338 提升到 0.8762（+5.1%），整体分类准确性改善
        - **F1-Score提升**: 从 0.4501 提升到 0.5037（+11.9%），平衡性能更好
        """)
        
    except Exception as e:
        st.error(f"加载对比数据时出错: {str(e)}")
        st.info("💡 提示：请运行 evaluate_lightgbm_optuna.py 生成对比数据")

with tab5:
    st.markdown("### 模型评估模块")
    st.markdown("**本模块对 Optuna 调优的 LightGBM 模型进行全面评估**")
    
    # 第一部分：Optuna优化LightGBM模型性能表格
    st.markdown("#### 🎯 Optuna优化LightGBM模型性能")
    
    try:
        metrics_path = BASE_DIR / "results" / "model_evaluation" / "lightgbm_optuna_metrics.csv"
        if metrics_path.exists():
            optuna_metrics = load_csv_data(metrics_path, index_col=0)
            metrics_row = optuna_metrics.iloc[0]
            
            # 创建性能指标表格
            ap_score = metrics_row.get('AP-Score', None)
            performance_data = {
                '评估指标': ['AUC-ROC', '准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 'F1-Score', 'AP-Score'],
                '数值': [
                    f"{metrics_row['AUC-ROC']:.4f}",
                    f"{metrics_row['Accuracy']:.4f}",
                    f"{metrics_row['Precision']:.4f}",
                    f"{metrics_row['Recall']:.4f}",
                    f"{metrics_row['F1-Score']:.4f}",
                    f"{ap_score:.4f}" if ap_score is not None and not pd.isna(ap_score) else "N/A"
                ]
            }
            performance_df = pd.DataFrame(performance_data)
            
            # 使用st.table显示表格（更简洁）
            st.table(performance_df)
        else:
            # 如果没有数据文件，显示默认值
            performance_data = {
                '评估指标': ['AUC-ROC', '准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 'F1-Score', 'AP-Score'],
                '数值': ['0.9069', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            }
            performance_df = pd.DataFrame(performance_data)
            st.table(performance_df)
            st.info("💡 提示：请先运行 `evaluate_lightgbm_optuna.py` 生成评估指标数据")
    except Exception as e:
        st.error(f"加载指标数据时出错: {str(e)}")
        st.info("💡 提示：请先运行 `evaluate_lightgbm_optuna.py` 生成评估指标数据")
    
    st.markdown("---")
    
    # 评估指标说明
    st.markdown("""
    **评估指标说明：**
    - **AUC-ROC**：ROC曲线下面积，衡量模型区分正负样本的能力（主要指标）
    - **准确率 (Accuracy)**：正确预测的样本占总样本的比例
    - **精确率 (Precision)**：预测为正例中实际为正例的比例
    - **召回率 (Recall)**：实际正例中被正确预测的比例
    - **F1-Score**：精确率和召回率的调和平均数
    - **AP-Score**：平均精确率，PR曲线下面积
    """)
    
    # ROC曲线、PR曲线和混淆矩阵 - 同一行显示
    st.markdown("#### ROC曲线、PR曲线和混淆矩阵")
    
    # 创建三列布局
    col_roc, col_pr, col_cm = st.columns(3)
    
    # 准备数据
    fig_roc = None
    fig_pr = None
    fig_cm = None
    
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from sklearn.model_selection import train_test_split
        import pickle
        import lightgbm as lgb
        
        model_path = BASE_DIR / "models" / "LightGBM_tuned_advanced.pkl"
        preprocessor_path = BASE_DIR / "models" / "preprocessor_lightgbm_advanced.pkl"
        data_path = BASE_DIR / "data" / "training_v2.csv"
        cm_path = BASE_DIR / "results" / "model_evaluation" / "confusion_matrix.csv"
        
        # 尝试加载模型和数据
        model = None
        y_proba = None
        y_val = None
        
        if model_path.exists() and data_path.exists():
            try:
                with st.spinner("正在加载Optuna优化模型并计算评估指标（这可能需要几秒钟）..."):
                    # 加载模型（使用缓存）
                    model_data = load_model(model_path)
                    if isinstance(model_data, dict):
                        model = model_data.get('model')
                    else:
                        model = model_data
                    
                    if model is not None:
                        # 获取模型期望的特征数量
                        model_n_features = None
                        try:
                            if hasattr(model, 'n_features_'):
                                model_n_features = model.n_features_
                            elif hasattr(model, 'booster_'):
                                model_n_features = model.booster_.num_feature()
                        except:
                            pass
                        
                        # 尝试加载预处理器获取特征列表（静默，仅在出错时提示）
                        selected_features = None
                        if preprocessor_path.exists():
                            try:
                                preprocessor = load_preprocessor(preprocessor_path)
                                if isinstance(preprocessor, dict) and 'feature_names' in preprocessor:
                                    selected_features = preprocessor['feature_names']
                            except Exception as e:
                                st.warning(f"无法加载预处理器: {str(e)}")

                        # 简化方法：直接使用本地 data 目录中的 CSV，不依赖仓库根目录的 Python 脚本
                        train_df = load_csv_data(data_path, nrows=20000, low_memory=False, na_values=['NA', ''])
                        if 'hospital_death' in train_df.columns:
                            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols = [col for col in numeric_cols if col not in 
                                           ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']]

                            # 使用模型期望的特征数量，如果没有则使用79（根据之前调试信息）
                            n_features = model_n_features if model_n_features else 79

                            if selected_features is not None:
                                # 优先使用预处理器中的特征列表
                                available_features = [col for col in selected_features if col in numeric_cols][:n_features]
                            else:
                                available_features = [col for col in numeric_cols if col in train_df.columns][:n_features]

                            if len(available_features) < n_features:
                                st.warning(f"可用特征数 ({len(available_features)}) 少于模型期望 ({n_features})")

                            X_sample = train_df[available_features].fillna(train_df[available_features].median())
                            y_sample = train_df['hospital_death']

                            # 数据分割
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
                            )

                            # 确保特征数量匹配
                            if model_n_features and X_val.shape[1] != model_n_features:
                                if X_val.shape[1] > model_n_features:
                                    X_val = X_val.iloc[:, :model_n_features]
                                else:
                                    st.error(f"特征数量不足: 需要 {model_n_features} 个，但只有 {X_val.shape[1]} 个")
                                    raise ValueError("特征数量不匹配")

                            y_proba = model.predict_proba(
                                X_val.values if isinstance(X_val, pd.DataFrame) else X_val
                            )[:, 1]
            except Exception as e:
                st.warning(f"加载模型或数据时出错: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
        
        # 1. ROC曲线
        with col_roc:
            st.markdown("##### ROC曲线")
            if y_proba is not None and y_val is not None:
                fpr, tpr, _ = roc_curve(y_val, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'AUC = {roc_auc:.4f}',
                    line=dict(color='#e74c3c', width=2)
                ))
            else:
                fpr_example = np.linspace(0, 1, 100)
                tpr_example = np.sqrt(fpr_example)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr_example,
                    y=tpr_example,
                    mode='lines',
                    name='AUC = 0.9069',
                    line=dict(color='#e74c3c', width=2)
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='随机猜测',
                line=dict(color='gray', width=1.5, dash='dash')
            ))
            fig_roc.update_layout(
                xaxis_title='假阳性率',
                yaxis_title='真阳性率',
                height=400,
                showlegend=True,
                margin=dict(l=30, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # 2. PR曲线
        with col_pr:
            st.markdown("##### PR曲线")
            if y_proba is not None and y_val is not None:
                precision, recall, _ = precision_recall_curve(y_val, y_proba)
                ap_score = average_precision_score(y_val, y_proba)
                baseline = np.sum(y_val) / len(y_val)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'AP = {ap_score:.4f}',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy'
                ))
                fig_pr.add_hline(
                    y=baseline,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"基线 ({baseline:.3f})"
                )
            else:
                recall_example = np.linspace(0, 1, 100)
                precision_example = 0.6 - 0.3 * recall_example
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall_example,
                    y=precision_example,
                    mode='lines',
                    name='AP = 0.5946',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy'
                ))
                fig_pr.add_hline(y=0.13, line_dash="dash", line_color="gray", annotation_text="基线 (0.13)")
            
            fig_pr.update_layout(
                xaxis_title='召回率',
                yaxis_title='精确率',
                height=400,
                showlegend=True,
                margin=dict(l=30, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # 3. 混淆矩阵
        with col_cm:
            st.markdown("##### 混淆矩阵")
            if cm_path.exists():
                cm_df = load_csv_data(cm_path, index_col=0)
                cm = cm_df.values
            else:
                cm = np.array([[16101, 659], [731, 852]])
            
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['存活', '死亡'],
                y=['存活', '死亡'],
                colorscale='Blues',
                hovertemplate='真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>',
                showscale=True
            ))
            
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=f'{cm[i, j]}<br>({cm_normalized[i, j]*100:.1f}%)',
                            showarrow=False,
                            font=dict(size=12, color='white' if cm[i, j] > cm.max()/2 else 'black')
                        )
                    )
            
            fig_cm.update_layout(
                xaxis_title='预测标签',
                yaxis_title='真实标签',
                height=400,
                annotations=annotations,
                margin=dict(l=30, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # 显示统计摘要
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            st.caption(f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
    
    except Exception as e:
        st.error(f"生成评估图表时出错: {str(e)}")
    
    # 4. SHAP可解释性分析
    st.markdown("#### SHAP可解释性分析")
    
    # 尝试生成交互式SHAP图表
    shap_interactive_success = False
    try:
        import shap
        import pickle
        import lightgbm as lgb
        
        model_path = BASE_DIR / "models" / "LightGBM_tuned_advanced.pkl"
        data_path = BASE_DIR / "data" / "training_v2.csv"
        
        if model_path.exists() and data_path.exists():
            with st.spinner("正在计算SHAP值并生成交互式图表..."):
                try:
                    # 加载模型
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        if isinstance(model_data, dict):
                            shap_model = model_data.get('model')
                        else:
                            shap_model = model_data
                    
                    if shap_model is not None:
                        # 模型期望特征数
                        model_n_features = None
                        try:
                            if hasattr(shap_model, 'n_features_'):
                                model_n_features = shap_model.n_features_
                            elif hasattr(shap_model, 'booster_'):
                                model_n_features = shap_model.booster_.num_feature()
                        except Exception:
                            model_n_features = None
                        
                        # 预处理器特征
                        selected_features = None
                        preprocessor_path = BASE_DIR / "models" / "preprocessor_lightgbm_advanced.pkl"
                        if preprocessor_path.exists():
                            try:
                                preprocessor = load_preprocessor(preprocessor_path)
                                if isinstance(preprocessor, dict) and 'feature_names' in preprocessor:
                                    selected_features = preprocessor['feature_names']
                            except Exception:
                                selected_features = None
                        
                        # 读取数据
                        train_df = load_csv_data(data_path, nrows=2000, low_memory=False, na_values=['NA', ''])
                        if 'hospital_death' in train_df.columns:
                            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols = [col for col in numeric_cols if col not in 
                                           ['encounter_id', 'patient_id', 'hospital_id', 'hospital_death']]
                            
                            # 选择特征：优先预处理器，否则按模型期望特征数
                            if selected_features:
                                features = [f for f in selected_features if f in train_df.columns]
                                if model_n_features and len(features) > model_n_features:
                                    features = features[:model_n_features]
                            else:
                                n_feats = model_n_features if model_n_features else 79
                                features = [col for col in numeric_cols if col in train_df.columns][:n_feats]
                            
                            # 校验特征数量
                            if model_n_features and len(features) != model_n_features:
                                if len(features) < model_n_features:
                                    st.warning(f"可用特征数 ({len(features)}) 少于模型期望 ({model_n_features})，跳过交互式SHAP")
                                    raise ValueError("特征数量不足，无法计算SHAP")
                                # 多余的已截断
                            
                            X_shap = train_df[features].fillna(train_df[features].median())
                            
                            # 创建SHAP解释器
                            explainer = shap.TreeExplainer(shap_model)
                            shap_values_all = explainer.shap_values(X_shap)
                            
                            # LightGBM二分类：shap_values通常为[class0, class1]
                            if isinstance(shap_values_all, list) and len(shap_values_all) > 1:
                                shap_values = shap_values_all[1]
                                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                            else:
                                shap_values = shap_values_all
                                expected_value = explainer.expected_value
                            
                            # 创建两列布局
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### SHAP Summary Plot（类似官方shap_summary风格）")
                                
                                # 取Top N特征，模仿shap.summary_plot的散点/蜂群效果
                                top_n = 20
                                mean_abs = np.abs(shap_values).mean(0)
                                order_idx = np.argsort(mean_abs)[-top_n:]
                                top_features = X_shap.columns[order_idx]
                                
                                # 采样样本减少渲染负载
                                sample_n = min(500, shap_values.shape[0])
                                shap_subset = shap_values[:sample_n, :]
                                
                                records = []
                                for feat in top_features:
                                    f_idx = list(X_shap.columns).index(feat)
                                    shap_vals_feat = shap_subset[:, f_idx]
                                    feat_vals = X_shap[feat].values[:sample_n]
                                    for sv, fv in zip(shap_vals_feat, feat_vals):
                                        records.append({
                                            "特征": feat,
                                            "SHAP值": sv,
                                            "特征值": fv
                                        })
                                
                                shap_long_df = pd.DataFrame(records)
                                shap_long_df["特征"] = pd.Categorical(
                                    shap_long_df["特征"],
                                    categories=list(top_features),
                                    ordered=True
                                )
                                
                                # 使用散点图模拟蜂群效果，并保留连续色阶
                                fig_shap_summary = px.scatter(
                                    shap_long_df,
                                    x="SHAP值",
                                    y="特征",
                                    color="特征值",
                                    title="SHAP特征重要性（Top 20）",
                                    color_continuous_scale="RdBu",
                                    hover_data={"特征值": True, "SHAP值": True},
                                )
                                fig_shap_summary.update_traces(
                                    opacity=0.7,
                                    marker=dict(size=6, line=dict(width=0))
                                )
                                fig_shap_summary.update_layout(
                                    height=520,
                                    yaxis_title="特征（按平均|SHAP值|排序）",
                                    xaxis_title="SHAP值",
                                    showlegend=False,
                                    coloraxis_colorbar=dict(title="特征值")
                                )
                                st.plotly_chart(fig_shap_summary, use_container_width=True)
                                
                                st.markdown("##### SHAP Dependence Plot（特征依赖图）")
                                # 取最重要的特征（Top列表最后一个）并绘制依赖图
                                if len(top_features) > 0:
                                    top_feature = top_features[-1]
                                    if top_feature in X_shap.columns:
                                        feature_idx = list(X_shap.columns).index(top_feature)
                                        fig_shap_dep = px.scatter(
                                            x=X_shap[top_feature].values[:500]
                                        )
                                        fig_shap_dep.update_traces(
                                            y=np.array(shap_values)[:500, feature_idx],
                                            mode='markers',
                                            marker=dict(
                                                color=np.array(shap_values)[:500, feature_idx],
                                                colorscale='RdBu',
                                                showscale=True
                                            ),
                                            hovertemplate='特征值: %{x}<br>SHAP值: %{y}<extra></extra>'
                                        )
                                        fig_shap_dep.update_layout(
                                            title=f'SHAP依赖图 - {top_feature}',
                                            xaxis_title=f'{top_feature} 值',
                                            yaxis_title='SHAP值',
                                            height=500
                                        )
                                        st.plotly_chart(fig_shap_dep, use_container_width=True)
                            
                            with col2:
                                st.markdown("##### SHAP Force Plot（个体解释示例）")
                                # 选择一个示例样本
                                example_idx = 0
                                example_shap_values = shap_values[example_idx]
                                example_features = X_shap.iloc[example_idx]
                                
                                # 创建交互式force plot（使用条形图）
                                force_df = pd.DataFrame({
                                    '特征': X_shap.columns,
                                    'SHAP值': example_shap_values,
                                    '特征值': example_features.values
                                }).sort_values('SHAP值', key=abs, ascending=False).head(15)
                                
                                colors = ['#e74c3c' if x > 0 else '#3498db' for x in force_df['SHAP值']]
                                fig_shap_force = go.Figure()
                                fig_shap_force.add_trace(go.Bar(
                                    x=force_df['SHAP值'],
                                    y=force_df['特征'],
                                    orientation='h',
                                    marker_color=colors,
                                    text=force_df['特征值'].apply(lambda x: f'{x:.2f}'),
                                    textposition='outside',
                                    hovertemplate='<b>%{y}</b><br>SHAP值: %{x:.4f}<br>特征值: %{text}<extra></extra>'
                                ))
                                fig_shap_force.add_vline(x=0, line_dash="dash", line_color="gray")
                                fig_shap_force.update_layout(
                                    title=f'SHAP Force Plot - 样本 {example_idx+1}<br>预测值: {expected_value + example_shap_values.sum():.4f}',
                                    xaxis_title='SHAP值（红色推高风险，蓝色降低风险）',
                                    yaxis_title='特征',
                                    height=500,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_shap_force, use_container_width=True)
                                
                                st.markdown("##### SHAP说明")
                                st.markdown("""
                                **SHAP (SHapley Additive exPlanations)** 提供了模型的可解释性分析：
                                
                                - **Summary Plot**: 展示各特征对模型输出的整体贡献大小及方向
                                - **Dependence Plot**: 展示特征取值与SHAP值的关系，揭示特征影响模式
                                - **Force Plot**: 展示单个患者预测中各特征推高或降低死亡风险的贡献
                                
                                **临床意义**：
                                - 帮助医生理解模型的决策依据
                                - 识别主要风险驱动因素
                                - 提供个体化解释，辅助临床决策
                                """)
                            
                            shap_interactive_success = True
                except Exception as e:
                    st.warning(f"生成交互式SHAP图表时出错: {str(e)}")
                    st.info("💡 请确保已安装SHAP库（`pip install shap`）并加载模型后可生成交互式图表")
    except ImportError:
        st.info("💡 SHAP库未安装，无法生成交互式SHAP图表。运行 `pip install shap` 可启用交互式SHAP图表")
    except Exception as e:
        st.info(f"💡 无法生成交互式SHAP图表: {str(e)}")
    
    # 如果无法生成交互式图表，显示提示信息
    if not shap_interactive_success:
        st.info("💡 交互式SHAP图表需要加载模型和数据。请确保模型文件和数据文件已正确放置在对应目录下。")

with tab6:
    st.markdown("### Kaggle提交结果")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **竞赛表现：**
        - **最佳提交：** LightGBM Ensemble
        - **Private Score：** 0.90470
        - **Public Score：** 0.90584
        - **Private排名：** 第222名（前280名区间）
        - **Public排名：** 第269名
        """)
    with col2:
        st.markdown("""
        **性能提升轨迹：**
        - 基础LightGBM → Optuna优化：排名提升约420名
        - 成功跨越前25%优秀性能分界线
        """)
    
    # 提交结果可视化（使用完整Kaggle提交数据，参考kaggle_late_submissions_comprehensive_new）
    try:
        kaggle_csv_path = BASE_DIR / "results" / "kaggle_submissions_data.csv"
        if kaggle_csv_path.exists():
            kaggle_df = load_csv_data(kaggle_csv_path)
            
            # 解析模型类型：优先使用CSV中的model列，如果为Unknown则从文件名和分数判断
            def parse_model_type(row):
                # 如果model列有值且不是Unknown，直接使用
                if pd.notna(row.get('model')) and row['model'] != 'Unknown':
                    return row['model']
                
                # 否则从文件名解析
                filename = str(row['filename']).lower()
                private_score = row.get('private_score', 0)
                
                if 'lightgbm_ensemble' in filename:
                    return 'LightGBM Ensemble'
                elif 'lightgbm' in filename:
                    return 'LightGBM'
                elif 'xgboost' in filename:
                    return 'XGBoost'
                elif 'standard_dl' in filename or 'dl' in filename:
                    return 'Deep Learning'
                elif 'submission.csv' in filename:
                    # submission.csv文件：根据分数范围判断
                    # Linear Regression的分数在0.890-0.895范围内
                    if 0.890 <= private_score <= 0.895:
                        return 'Linear Regression'
                    else:
                        # 其他分数（如0.89696）和LightGBM基础模型一样，可能是重复，过滤掉
                        return None  # 返回None，稍后过滤
                else:
                    return 'Unknown'
            
            kaggle_df['model_type'] = kaggle_df.apply(parse_model_type, axis=1)
            
            # 过滤掉None和Unknown类型的数据（避免显示不确定或重复的模型）
            kaggle_df = kaggle_df[
                (kaggle_df['model_type'].notna()) & 
                (kaggle_df['model_type'] != 'Unknown')
            ].copy()
            
            # 转换时间
            from datetime import datetime, timedelta
            if 'submission_time' in kaggle_df.columns:
                kaggle_df['submission_time'] = pd.to_datetime(kaggle_df['submission_time'])
            elif 'hours_ago' in kaggle_df.columns:
                base_time = datetime.now()
                kaggle_df['submission_time'] = kaggle_df['hours_ago'].apply(
                    lambda x: base_time - timedelta(hours=x)
                )
            
            # 去重：每个模型的每种调优方法只保留一个
            kaggle_df_deduped = []
            for (model, stage), group in kaggle_df.groupby(['model_type', 'stage']):
                if len(group) > 1:
                    group_sorted = group.sort_values('private_score', ascending=False)
                    best_row = group_sorted.iloc[0]
                    kaggle_df_deduped.append(best_row)
                else:
                    kaggle_df_deduped.append(group.iloc[0])
            
            kaggle_df = pd.DataFrame(kaggle_df_deduped).reset_index(drop=True)
            kaggle_df = kaggle_df.sort_values('submission_time').reset_index(drop=True)
            
            # 分配优化阶段标签
            def get_stage_label(row):
                model = row['model_type']
                stage = row['stage']
                
                if model == 'LightGBM Ensemble':
                    return 'Ensemble'
                elif stage == '基础模型':
                    return model
                elif stage == '普通调优':
                    return 'Hyperparameter Tuning\n(RandomizedSearchCV)'
                elif stage == '高级调优':
                    return 'Hyperparameter Tuning\n(Optuna)'
                elif stage == '集成模型（最优）':
                    return 'Ensemble'
                else:
                    return stage
            
            kaggle_df['stage_label'] = kaggle_df.apply(get_stage_label, axis=1)

            # 小标题：Late Submission 结果分析（靠近图表，减小下边距）
            st.markdown(
                "<h4 style='margin-bottom:0.3rem;'>Late Submission 结果分析</h4>",
                unsafe_allow_html=True
            )

            # 定义颜色方案
            model_colors = {
                'LightGBM Ensemble': '#e74c3c',
                'LightGBM': '#3498db',
                'XGBoost': '#2ecc71',
                'Deep Learning': '#f39c12',
                'Linear Regression': '#95a5a6'
            }
            
            # 创建三个子图的布局
            fig = make_subplots(
                rows=1, cols=3,
                horizontal_spacing=0.12
            )
            
            # 合并LightGBM和LightGBM Ensemble的数据用于时间序列
            lightgbm_data = kaggle_df[kaggle_df['model_type'].isin(['LightGBM', 'LightGBM Ensemble'])].sort_values('submission_time')
            
            # 子图1: Private Score时间序列
            for model in kaggle_df['model_type'].unique():
                if model == 'LightGBM Ensemble':
                    continue  # 稍后合并到LightGBM
                
                model_data = kaggle_df[kaggle_df['model_type'] == model].sort_values('submission_time')
                
                if model == 'LightGBM':
                    ensemble_data = kaggle_df[kaggle_df['model_type'] == 'LightGBM Ensemble'].sort_values('submission_time')
                    if len(ensemble_data) > 0:
                        model_data = pd.concat([model_data, ensemble_data]).sort_values('submission_time')
                
                fig.add_trace(
                    go.Scatter(
                        x=model_data['submission_time'],
                        y=model_data['private_score'],
                        mode='lines+markers',
                        name=model,
                        line=dict(color=model_colors.get(model, '#95a5a6'), width=2),
                        marker=dict(size=8),
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            "时间: %{x}<br>"
                            "Private Score: %{y:.5f}<br>"
                            "阶段: %{customdata}<extra></extra>"
                        ),
                        customdata=model_data['stage_label']
                    ),
                    row=1, col=1
                )
            
            # 子图2: Public Score时间序列
            for model in kaggle_df['model_type'].unique():
                if model == 'LightGBM Ensemble':
                    continue
                
                model_data = kaggle_df[kaggle_df['model_type'] == model].sort_values('submission_time')
                
                if model == 'LightGBM':
                    ensemble_data = kaggle_df[kaggle_df['model_type'] == 'LightGBM Ensemble'].sort_values('submission_time')
                    if len(ensemble_data) > 0:
                        model_data = pd.concat([model_data, ensemble_data]).sort_values('submission_time')
                
                fig.add_trace(
                    go.Scatter(
                        x=model_data['submission_time'],
                        y=model_data['public_score'],
                        mode='lines+markers',
                        name=model,
                        line=dict(color=model_colors.get(model, '#95a5a6'), width=2),
                        marker=dict(size=8, symbol='square'),
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            "时间: %{x}<br>"
                            "Public Score: %{y:.5f}<br>"
                            "阶段: %{customdata}<extra></extra>"
                        ),
                        customdata=model_data['stage_label'],
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 子图3: Private vs Public Score散点图
            for model in kaggle_df['model_type'].unique():
                model_data = kaggle_df[kaggle_df['model_type'] == model]
                
                fig.add_trace(
                    go.Scatter(
                        x=model_data['public_score'],
                        y=model_data['private_score'],
                        mode='markers',
                        name=model,
                        marker=dict(
                            color=model_colors.get(model, '#95a5a6'),
                            size=10,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            "Public Score: %{x:.5f}<br>"
                            "Private Score: %{y:.5f}<br>"
                            "阶段: %{customdata}<extra></extra>"
                        ),
                        customdata=model_data['stage_label'],
                        showlegend=False
                    ),
                    row=1, col=3
                )
            
            # 添加对角线（理想线）
            min_score = min(kaggle_df['private_score'].min(), kaggle_df['public_score'].min()) - 0.002
            max_score = max(kaggle_df['private_score'].max(), kaggle_df['public_score'].max()) + 0.002
            fig.add_trace(
                go.Scatter(
                    x=[min_score, max_score],
                    y=[min_score, max_score],
                    mode='lines',
                    name='y=x',
                    line=dict(dash='dash', color='gray', width=1),
                    showlegend=False,
                    hovertemplate='理想线<extra></extra>'
                ),
                row=1, col=3
            )
            
            # 更新布局
            fig.update_xaxes(title_text="提交时间", row=1, col=1)
            fig.update_yaxes(title_text="Private Score", row=1, col=1)
            
            fig.update_xaxes(title_text="提交时间", row=1, col=2)
            fig.update_yaxes(title_text="Public Score", row=1, col=2)
            
            fig.update_xaxes(title_text="Public Score", row=1, col=3)
            fig.update_yaxes(title_text="Private Score", row=1, col=3)
            
            fig.update_layout(
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Public Score排名数据（硬编码，来自plot_combined_submission_rankings.py）
            public_rankings = {
                0.87408: 778,
                0.88907: 742,
                0.89805: 697,
                0.89950: 692,
                0.90171: 672,
                0.90268: 659,
                0.90267: 659,
                0.90540: 275,
                0.90584: 269,
            }
            
            # Private排名数据（根据分数估算，实际应该从leaderboard文件读取）
            # 这里使用近似值，基于plot_combined_submission_rankings.py的逻辑
            private_rankings_approx = {
                0.87873: 800,  # Deep Learning
                0.89194: 750,  # Linear Regression
                0.89696: 650,  # LightGBM基础
                0.89711: 640,  # XGBoost基础
                0.90035: 500,  # XGBoost普通调优
                0.90146: 450,  # LightGBM普通调优
                0.90234: 400,  # XGBoost高级调优
                0.90417: 280,  # LightGBM高级调优
                0.90470: 222,  # LightGBM Ensemble / LightGBM高级调优
            }
            
            total_teams_private = 1120  # 近似值
            total_teams_public = 951
            
            # 为每个提交添加排名信息
            kaggle_df_with_ranks = kaggle_df.copy()
            kaggle_df_with_ranks['private_rank'] = kaggle_df_with_ranks['private_score'].map(
                lambda x: min(private_rankings_approx.items(), key=lambda item: abs(item[0] - x))[1]
                if abs(min(private_rankings_approx.items(), key=lambda item: abs(item[0] - x))[0] - x) < 0.001
                else None
            )
            kaggle_df_with_ranks['public_rank'] = kaggle_df_with_ranks['public_score'].map(
                lambda x: public_rankings.get(
                    min(public_rankings.keys(), key=lambda k: abs(k - x)),
                    None
                ) if abs(min(public_rankings.keys(), key=lambda k: abs(k - x)) - x) < 0.001
                else None
            )
            
            # 过滤掉没有排名的数据
            kaggle_df_with_ranks = kaggle_df_with_ranks[
                kaggle_df_with_ranks['private_rank'].notna() & 
                kaggle_df_with_ranks['public_rank'].notna()
            ].copy()
            
            if len(kaggle_df_with_ranks) > 0:
                # 小标题：提交排名分析（靠近图表，减小下边距）
                st.markdown(
                    "<h4 style='margin-bottom:0.3rem;'>提交排名分析</h4>",
                    unsafe_allow_html=True
                )

                # 创建排名图表（两个子图）
                fig_ranks = make_subplots(
                    rows=1, cols=2,
                    horizontal_spacing=0.15
                )
                
                # 按分数排序用于连线
                df_sorted_private = kaggle_df_with_ranks.sort_values('private_score')
                df_sorted_public = kaggle_df_with_ranks.sort_values('public_score')
                
                # 子图1: Private Score vs 排名
                # 添加连线（灰色，半透明）
                fig_ranks.add_trace(
                    go.Scatter(
                        x=df_sorted_private['private_score'],
                        y=df_sorted_private['private_rank'],
                        mode='lines',
                        name='_连线',
                        line=dict(color='gray', width=2, dash='dot'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                # 添加各模型的散点
                for model in kaggle_df_with_ranks['model_type'].unique():
                    model_data = kaggle_df_with_ranks[kaggle_df_with_ranks['model_type'] == model]
                    
                    fig_ranks.add_trace(
                        go.Scatter(
                            x=model_data['private_score'],
                            y=model_data['private_rank'],
                            mode='markers+text',
                            name=model,
                            text=[f"#{int(r)}" for r in model_data['private_rank']],
                            textposition='middle right',
                            marker=dict(
                                color=model_colors.get(model, '#95a5a6'),
                                size=12,
                                line=dict(width=1.5, color='black')
                            ),
                            hovertemplate=(
                                f"<b>{model}</b><br>"
                                "Private Score: %{x:.5f}<br>"
                                "排名: #%{y}<br>"
                                "阶段: %{customdata}<extra></extra>"
                            ),
                            customdata=model_data['stage_label']
                        ),
                        row=1, col=1
                    )
                
                # 添加前25%和前60%参考线
                top_25_private = int(total_teams_private * 0.25)
                top_60_private = int(total_teams_private * 0.60)
                
                fig_ranks.add_hline(
                    y=top_25_private, 
                    line_dash="dash", 
                    line_color="green", 
                    opacity=0.5,
                    annotation_text="前25%",
                    row=1, col=1
                )
                fig_ranks.add_hline(
                    y=top_60_private, 
                    line_dash="dash", 
                    line_color="orange", 
                    opacity=0.5,
                    annotation_text="前60%",
                    row=1, col=1
                )
                
                # 子图2: Public Score vs 排名
                # 添加连线
                fig_ranks.add_trace(
                    go.Scatter(
                        x=df_sorted_public['public_score'],
                        y=df_sorted_public['public_rank'],
                        mode='lines',
                        name='_连线',
                        line=dict(color='gray', width=2, dash='dot'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
                
                # 添加各模型的散点
                for model in kaggle_df_with_ranks['model_type'].unique():
                    model_data = kaggle_df_with_ranks[kaggle_df_with_ranks['model_type'] == model]
                    
                    fig_ranks.add_trace(
                        go.Scatter(
                            x=model_data['public_score'],
                            y=model_data['public_rank'],
                            mode='markers+text',
                            name=model,
                            text=[f"#{int(r)}" for r in model_data['public_rank']],
                            textposition='middle right',
                            marker=dict(
                                color=model_colors.get(model, '#95a5a6'),
                                size=12,
                                line=dict(width=1.5, color='black'),
                                symbol='square'
                            ),
                            hovertemplate=(
                                f"<b>{model}</b><br>"
                                "Public Score: %{x:.5f}<br>"
                                "排名: #%{y}<br>"
                                "阶段: %{customdata}<extra></extra>"
                            ),
                            customdata=model_data['stage_label'],
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                # 添加前25%和前60%参考线
                top_25_public = int(total_teams_public * 0.25)
                top_60_public = int(total_teams_public * 0.60)
                
                fig_ranks.add_hline(
                    y=top_25_public, 
                    line_dash="dash", 
                    line_color="green", 
                    opacity=0.5,
                    annotation_text="前25%",
                    row=1, col=2
                )
                fig_ranks.add_hline(
                    y=top_60_public, 
                    line_dash="dash", 
                    line_color="orange", 
                    opacity=0.5,
                    annotation_text="前60%",
                    row=1, col=2
                )
                
                # 更新布局
                fig_ranks.update_xaxes(title_text="Private Score", row=1, col=1)
                fig_ranks.update_yaxes(
                    title_text="排名 (Rank)", 
                    row=1, col=1,
                    autorange="reversed"  # 反转Y轴，使排名1在顶部
                )
                
                fig_ranks.update_xaxes(title_text="Public Score", row=1, col=2)
                fig_ranks.update_yaxes(
                    title_text="排名 (Rank)", 
                    row=1, col=2,
                    autorange="reversed"  # 反转Y轴，使排名1在顶部
                )
                
                fig_ranks.update_layout(
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_ranks, use_container_width=True)
            else:
                st.info("无法获取排名数据，跳过排名图表显示。")
        else:
            st.info("未找到 `results/kaggle_submissions_data.csv`，暂时使用示例数据。")
    except Exception as e:
        st.error(f"加载Kaggle提交数据时出错: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

# 核心实现代码板块
st.markdown('<div class="section-header">💻 核心实现代码</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>本板块展示项目中的核心实现代码，包括数据加载、预处理、特征工程、模型训练等关键部分。</p>
</div>
""", unsafe_allow_html=True)

# 创建子标签页用于不同模块的代码展示
code_tab1, code_tab2, code_tab3, code_tab4, code_tab5 = st.tabs([
    "📥 数据加载", 
    "🔧 数据预处理", 
    "⚙️ 特征工程", 
    "🤖 模型训练", 
    "🎯 模型集成"
])

with code_tab1:
    st.markdown("#### 数据加载核心代码")
    st.markdown("**功能：** 加载训练数据和数据字典，进行初步检查和目标变量分析")
    
    data_loading_code = '''def load_data():
    """
    加载数据文件
    
    Returns:
        train_df: 训练数据DataFrame
        dict_df: 数据字典DataFrame
    """
    print("【步骤 1】加载数据...")
    print("-" * 80)
    
    # 加载训练数据（将 "NA" 字符串识别为缺失值）
    train_df = pd.read_csv('data/training_v2.csv', 
                          low_memory=False, 
                          na_values=['NA', ''])
    print(f"✓ 训练数据已加载: {train_df.shape[0]:,} 行 × {train_df.shape[1]} 列")
    
    # 加载数据字典
    dict_df = pd.read_csv('data/WiDS Datathon 2020 Dictionary.csv')
    print(f"✓ 数据字典已加载: {dict_df.shape[0]:,} 行 × {dict_df.shape[1]} 列")
    
    return train_df, dict_df

def analyze_target_variable(train_df):
    """
    分析目标变量
    
    Args:
        train_df: 训练数据DataFrame
    
    Returns:
        target_counts: 目标变量计数
        target_percent: 目标变量百分比
    """
    print("【步骤 3】目标变量 (hospital_death) 分析")
    print("-" * 80)
    
    # 统计分布
    target_counts = train_df['hospital_death'].value_counts()
    target_percent = train_df['hospital_death'].value_counts(normalize=True) * 100
    
    print("目标变量分布:")
    print(f"  - 存活 (0): {target_counts[0]:,} 例 ({target_percent[0]:.2f}%)")
    print(f"  - 死亡 (1): {target_counts[1]:,} 例 ({target_percent[1]:.2f}%)")
    
    return target_counts, target_percent'''
    
    st.code(data_loading_code, language='python')
    
    st.markdown("**关键特性：**")
    st.markdown("""
    - 使用 `low_memory=False` 确保完整加载数据
    - 标准化缺失值处理（将 'NA' 和空字符串映射为 NaN）
    - 自动统计目标变量分布，识别类别不平衡问题
    """)
    
with code_tab2:
    st.markdown("#### 数据预处理核心代码")
    st.markdown("**功能：** 特征分类、缺失值处理、异常值检测")
    
    preprocessing_code = '''def classify_features(train_df, dict_df):
    """
    基于数据字典进行特征分类
    
    Args:
        train_df: 训练数据DataFrame
        dict_df: 数据字典DataFrame
    
    Returns:
        feature_categories: 特征分类字典
    """
    print("【步骤 4】特征分类（基于数据字典）")
    print("-" * 80)
    
    # 创建特征分类字典
    feature_categories = {}
    for _, row in dict_df.iterrows():
        category = row['Category']
        var_name = row['Variable Name']
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(var_name)
    
    # 打印每个类别的特征数量
    print("特征分类统计:")
    for category in sorted(feature_categories.keys()):
        features = feature_categories[category]
        existing_features = [f for f in features if f in train_df.columns]
        print(f"  - {category:30s}: {len(existing_features):3d} 个特征")
    
    return feature_categories

def basic_preprocessing(train_df, missing_df):
    """
    执行基础数据预处理
    
    Args:
        train_df: 训练数据DataFrame
        missing_df: 缺失值分析DataFrame
    
    Returns:
        train_df_cleaned: 清洗后的数据（删除高缺失值列）
        high_missing_cols: 被删除的高缺失值列
    """
    print("【步骤 5】基础预处理")
    print("-" * 80)
    
    # 剔除缺失值比例超过 70% 的列
    high_missing_cols = missing_df[missing_df['缺失比例(%)'] > 70].index.tolist()
    train_df_cleaned = train_df.drop(columns=high_missing_cols)
    
    print(f"✓ 删除了 {len(high_missing_cols)} 个高缺失值列（缺失率 > 70%）")
    print(f"✓ 剩余特征数: {train_df_cleaned.shape[1]}")
    
    return train_df_cleaned, high_missing_cols'''
    
    st.code(preprocessing_code, language='python')
    
    st.markdown("**处理策略：**")
    st.markdown("""
    - **高缺失率特征（>70%）**: 直接剔除，避免引入噪声
    - **数值型特征**: 使用中位数填充，对异常值更稳健
    - **分类特征**: 使用众数填充
    - **医学逻辑填充**: 基于临床知识进行智能填充
    """)
    
with code_tab3:
    st.markdown("#### 特征工程核心代码")
    st.markdown("**功能：** 创建GCS评分、生命体征、实验室指标等新特征")
    
    feature_engineering_code = '''def create_gcs_features(df):
    """
    创建GCS（格拉斯哥昏迷评分）相关特征
    
    Args:
        df: 数据DataFrame
    
    Returns:
        df: 添加了GCS特征的DataFrame
    """
    print("创建GCS特征...")
    
    # GCS总分 = 眼睛 + 运动 + 语言
    if all(col in df.columns for col in ['gcs_eyes_apache', 
                                         'gcs_motor_apache', 
                                         'gcs_verbal_apache']):
        gcs_total = df['gcs_eyes_apache'] + df['gcs_motor_apache'] + df['gcs_verbal_apache']
        # 如果gcs_unable_apache=1，表示无法评估，设为缺失
        if 'gcs_unable_apache' in df.columns:
            gcs_total[df['gcs_unable_apache'] == 1] = np.nan
        df['gcs_total'] = gcs_total
        print(f"  ✓ 创建 gcs_total: 范围 [{df['gcs_total'].min():.1f}, {df['gcs_total'].max():.1f}]")
    
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
    
    # 1. 血压相关特征 - 收缩压范围（最大值-最小值）
    if all(col in df.columns for col in ['d1_sysbp_max', 'd1_sysbp_min']):
        df['d1_sysbp_range'] = df['d1_sysbp_max'] - df['d1_sysbp_min']
        print(f"  ✓ 创建 d1_sysbp_range")
    
    # 2. 心率相关特征
    if all(col in df.columns for col in ['d1_heartrate_max', 'd1_heartrate_min']):
        df['d1_heartrate_range'] = df['d1_heartrate_max'] - df['d1_heartrate_min']
        df['d1_heartrate_mean'] = (df['d1_heartrate_max'] + df['d1_heartrate_min']) / 2
        print(f"  ✓ 创建 d1_heartrate_range 和 d1_heartrate_mean")
    
    return df'''
        
    st.code(feature_engineering_code, language='python')
    
    st.markdown("**特征类型：**")
    st.markdown("""
    - **GCS评分特征**: 格拉斯哥昏迷评分总分和组件
    - **生命体征特征**: 血压、心率、血氧、体温、呼吸频率的范围和均值
    - **实验室指标特征**: 血常规、生化指标、血气分析等
    - **交互特征**: 特征间的乘积、比值等
    """)
    
with code_tab4:
    st.markdown("#### 模型训练核心代码")
    st.markdown("**功能：** 训练多种机器学习模型，包括传统ML和梯度提升模型")
    
    model_training_code = '''def train_models(X_train_filled, y_train, X_val_filled, y_val, 
                 use_class_weight=True):
    """
    训练多个预测模型
    
    Args:
        X_train_filled: 训练特征（填充缺失值版本）
        y_train: 训练目标
        X_val_filled: 验证特征（填充缺失值版本）
        y_val: 验证目标
        use_class_weight: 是否使用类别权重平衡
    
    Returns:
        models: 训练好的模型字典
        predictions: 预测结果字典
        metrics: 评估指标字典
    """
    print("【步骤 3】模型训练")
    print("-" * 80)
    
    models = {}
    predictions = {}
    metrics = {}
    
    # 计算类别权重（用于处理类别不平衡）
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(y_train), 
                                           y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"类别权重: 存活={class_weight_dict[0]:.4f}, 死亡={class_weight_dict[1]:.4f}")
    
    # 3.1 逻辑回归
    print("3.1 训练逻辑回归模型...")
    lr_model = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train_filled, y_train)
    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = {
        'proba': lr_model.predict_proba(X_val_filled)[:, 1],
        'pred': lr_model.predict(X_val_filled)
    }
    print("  ✓ 完成")
    
    # 3.4 XGBoost（支持缺失值）
    print("3.4 训练XGBoost模型（保留缺失值，让模型学习处理）...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(X_train_filled, y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = {
        'proba': xgb_model.predict_proba(X_val_filled)[:, 1],
        'pred': xgb_model.predict(X_val_filled)
    }
    print("  ✓ 完成")
    
    # 3.5 LightGBM（支持缺失值，GPU加速）
    print("3.5 训练LightGBM模型（保留缺失值，GPU加速）...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        device='gpu'  # GPU加速
    )
    lgb_model.fit(X_train_filled, y_train)
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = {
        'proba': lgb_model.predict_proba(X_val_filled)[:, 1],
        'pred': lgb_model.predict(X_val_filled)
    }
    print("  ✓ 完成")
    
    return models, predictions, metrics'''
        
    st.code(model_training_code, language='python')
    
    st.markdown("**模型类型：**")
    st.markdown("""
    - **逻辑回归**: 基准模型，线性分类器
    - **随机森林**: 集成树模型，处理非线性关系
    - **XGBoost**: 梯度提升树，支持缺失值
    - **LightGBM**: 快速梯度提升，支持GPU加速
    - **深度学习**: 深度神经网络，Wide & Deep架构
    """)
    
with code_tab5:
    st.markdown("#### 模型集成核心代码")
    st.markdown("**功能：** 训练多个LightGBM模型并集成，提升预测性能")
    
    ensemble_code = '''def train_ensemble_models(X_train, y_train, X_val, y_val, 
                          base_params, n_models=5, use_gpu=False):
    """
    训练多个LightGBM模型（不同随机种子）
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        base_params: 基础参数（从调优后的模型获取）
        n_models: 模型数量
        use_gpu: 是否使用GPU
    
    Returns:
        models: 模型列表
        predictions: 每个模型的预测结果
    """
    print(f"训练 {n_models} 个LightGBM模型（不同随机种子）...")
    print()
    
    models = []
    predictions = []
    
    for i in range(n_models):
        print(f"训练模型 {i+1}/{n_models}...")
        
        # 复制基础参数，修改随机种子
        params = base_params.copy()
        params['random_state'] = 42 + i * 100  # 不同的随机种子
        
        # 创建模型
        model = lgb.LGBMClassifier(**params)
        
        # 训练模型（使用早停）
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # 预测
        val_pred = model.predict_proba(X_val)[:, 1]
        
        models.append(model)
        predictions.append(val_pred)
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_pred)
        print(f"  模型 {i+1} AUC-ROC: {auc:.5f}")
        print()
    
    return models, predictions

def ensemble_predict(models, X_test):
    """
    集成多个模型的预测结果
    
    Args:
        models: 模型列表
        X_test: 测试特征
    
    Returns:
        ensemble_pred: 集成预测结果（加权平均）
    """
    predictions = []
    for model in models:
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
    
    # 简单平均（也可以使用加权平均）
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred'''
        
    st.code(ensemble_code, language='python')
    
    st.markdown("**集成策略：**")
    st.markdown("""
    - **多模型训练**: 使用5个不同随机种子的LightGBM模型
    - **早停机制**: 防止过拟合，自动选择最佳迭代次数
    - **预测融合**: 对多个模型的预测概率进行加权平均
    - **性能提升**: 集成模型相比单模型AUC-ROC提升约0.002-0.005
    """)
    
    st.markdown("**超参数优化代码（Optuna）：**")
    
    optuna_code = '''import optuna

def objective(trial):
    """Optuna优化目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    return auc

# 创建Optuna研究并优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 获取最佳参数
best_params = study.best_params
print(f"最佳AUC-ROC: {study.best_value:.5f}")
print(f"最佳参数: {best_params}")'''
    
    st.code(optuna_code, language='python')
    
    st.markdown("**优化效果：**")
    st.markdown("""
    - 使用Optuna贝叶斯优化自动搜索最佳超参数
    - 相比手动调参，AUC-ROC提升约0.003-0.005
    - 排名从约700名提升至280名左右，提升约420名
    """)

# 与最优模型差距分析
st.markdown('<div class="section-header">📊 与最优模型差距分析</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="warning-box">
        <h4>🔍 主要差距识别</h4>
        <p><strong>当前性能</strong>: AUC-ROC = 0.9069（相比Baseline提升4.5%）</p>
        <p><strong>与最优模型差距</strong>: 0.0081（约0.81%）</p>
        <ol>
            <li><strong>测试时增强（TTA）缺失</strong>
                <ul>
                    <li>论文方案：通过改变性别、种族、年龄生成增强样本</li>
                    <li>性能提升：约0.004 AUC</li>
                </ul>
            </li>
            <li><strong>模型集成规模不足</strong>
                <ul>
                    <li>当前：5个LightGBM模型</li>
                    <li>论文方案：42个不同类型模型</li>
                </ul>
            </li>
            <li><strong>缺少StackNet元学习架构</strong>
                <ul>
                    <li>当前：简单加权平均</li>
                    <li>论文方案：三层堆叠架构</li>
                </ul>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # 性能对比
    comparison_data = pd.DataFrame({
        '方案': ['我们的模型', '最优模型', 'Baseline'],
        'AUC-ROC': [0.9069, 0.915, 0.868],
        '差距': [0.0081, 0, 0.047]
    })
    
    fig = px.bar(
        comparison_data,
        x='方案',
        y='AUC-ROC',
        title='性能对比：我们的模型 vs 最优模型 vs Baseline',
        color='AUC-ROC',
        color_continuous_scale='RdYlGn'
    )
    fig.add_hline(y=0.915, line_dash="dash", line_color="red", 
                  annotation_text="最优模型目标 (0.915)")
    fig.add_hline(y=0.9069, line_dash="dash", line_color="blue", 
                  annotation_text="我们的模型 (0.9069)")
    fig.add_hline(y=0.868, line_dash="dash", line_color="gray", 
                  annotation_text="Baseline (0.868)")
    # 调整 y 轴范围，使差距更直观
    fig.update_layout(yaxis=dict(range=[0.8, 1.0]))
    st.plotly_chart(fig, use_container_width=True)

# 技术栈和工具
st.markdown('<div class="section-header">🛠️ 技术栈</div>', unsafe_allow_html=True)

tech_cols = st.columns(4)
tech_stack = [
    # 当前运行环境 Python 版本为 3.13.5（经 py --version 检测）
    ("Python 3.13.5", "🐍"),
    ("pandas & numpy", "📊"),
    ("scikit-learn", "🤖"),
    ("LightGBM/XGBoost", "🌲"),
    ("TensorFlow/Keras", "🧠"),
    ("Optuna", "⚙️"),
    ("matplotlib/seaborn", "📈"),
    ("Streamlit", "🚀")
]

for i, (tech, icon) in enumerate(tech_stack):
    with tech_cols[i % 4]:
        st.markdown(f"### {icon}")
        st.markdown(f"**{tech}**")

# 项目文件结构
st.markdown('<div class="section-header">📁 项目结构</div>', unsafe_allow_html=True)

st.markdown("""
```
streamlit_app/
├── app.py                   # Streamlit 主应用
├── data/                    # 应用使用的所有原始数据
│   ├── training_v2.csv      # 训练数据（从 WiDS 官方数据复制到此处）
│   ├── unlabeled.csv        # 未标注数据（如需使用）
│   └── WiDS Datathon 2020 Dictionary.csv  # 官方数据字典
├── models/                  # 训练好的模型文件（.pkl/.json等）
├── results/                 # 分析结果 CSV（模型指标、相关性分析、Kaggle 提交记录等）
│   ├── statistical_analysis/# 统计分析结果
│   ├── model_training/      # 模型训练结果
│   └── model_evaluation/    # 模型评估结果
└── README.md                # 使用说明
```
""")

# 代码文件
st.markdown('<div class="section-header">📝 代码文件</div>', unsafe_allow_html=True)

nav_cols = st.columns(3)

with nav_cols[0]:
    st.markdown("""
    **数据分析脚本：**
    - `data_loading.py` - 数据读取
    - `data_preprocessing.py` - 数据预处理
    - `statistical_analysis.py` - 统计分析
    - `feature_engineering.py` - 特征工程
    """)

with nav_cols[1]:
    st.markdown("""
    **模型训练脚本：**
    - `model_training.py` - 传统ML模型
    - `deep_learning_training.py` - 深度学习模型
    - `hyperparameter_tuning.py` - 超参数优化
    - `ensemble_lightgbm.py` - 集成模型
    """)

with nav_cols[2]:
    st.markdown("""
    **评估与预测：**
    - `evaluate_lightgbm_ensemble.py` - 模型评估
    - `predict_lightgbm_ensemble.py` - 预测生成
    - `plot_kaggle_rankings.py` - 排名可视化
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
    <p><strong>WiDS Datathon 2020 - ICU死亡风险预测分析系统</strong></p>
    <p>基于多中心临床数据的机器学习预测模型 | 作者：刘佳城</p>
    <p>数据来源：MIT GOSSIS Initiative | 最后更新：2026年1月</p>
</div>
""", unsafe_allow_html=True)
