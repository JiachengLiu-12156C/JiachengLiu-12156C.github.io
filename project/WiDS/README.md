# WiDS Datathon 2020 - Streamlit 数据分析与可视化应用

## 项目简介

这是一个基于 Streamlit 框架构建的交互式数据分析与可视化应用，用于展示 WiDS Datathon 2020 数据集的 ICU 死亡风险预测分析结果。

## 功能特性

- 📊 **数据概览**：展示数据集的基本信息和目标变量分布
- 📥 **数据读取模块**：展示数据加载和特征分类结果
- 🔧 **数据预处理模块**：展示缺失值处理和特征工程策略
- 📈 **统计分析模块**：展示描述性统计、相关性分析等结果
- 🤖 **模型训练模块**：展示不同模型的训练结果和性能对比
- 📊 **模型评估模块**：展示模型评估指标和最佳模型信息
- 🏆 **Kaggle结果**：展示竞赛提交结果和排名信息
- 💻 **核心实现代码**：展示关键代码实现，包括数据加载、预处理、特征工程、模型训练和集成等核心部分
- 📊 **差距分析**：对比当前方案与最优方案的差距

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

在项目根目录下运行：

```bash
streamlit run streamlit_app/app.py
```

或者：

```bash
cd streamlit_app
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址为：`http://localhost:8501`

## 项目结构（完全以 `streamlit_app` 为根目录）

```
streamlit_app/
├── app.py              # Streamlit 主应用文件
├── requirements.txt    # Python 依赖包
├── README.md           # 本文件
├── data/               # 应用使用的所有原始数据
│   ├── training_v2.csv
│   └── WiDS Datathon 2020 Dictionary.csv
├── models/             # 训练好的模型文件（如 LightGBM_tuned_advanced.pkl 等）
└── results/            # 分析结果 CSV（模型指标、统计分析、Kaggle 提交数据等）
    ├── statistical_analysis/  # 统计分析结果
    ├── model_training/       # 模型训练结果
    └── model_evaluation/     # 模型评估结果
```

## 注意事项

1. 确保数据文件位于 `streamlit_app/data/` 目录下（即与 `app.py` 同级的 `data` 文件夹）：
   - `training_v2.csv`
   - `WiDS Datathon 2020 Dictionary.csv`

2. 部分功能需要先在仓库根目录运行相应分析/训练脚本生成结果文件，然后**手动将生成的 CSV/模型复制到 `streamlit_app` 下对应子目录**：
   - 统计分析结果 CSV → 复制到 `streamlit_app/results/statistical_analysis/`
   - 模型训练与评估结果 CSV → 复制到 `streamlit_app/results/model_training/` 和 `streamlit_app/results/model_evaluation/`
   - 训练好的模型文件与预处理器 → 复制到 `streamlit_app/models/`
   - Kaggle 提交记录 CSV（如 `kaggle_submissions_data.csv`）→ 放在 `streamlit_app/results/`
   
   注意：应用使用交互式图表（Plotly）展示所有可视化内容，无需静态图片文件。

3. 如果某些图表或数据无法显示，请检查：
   - 数据文件是否存在
   - 结果文件是否已生成
   - 文件路径是否正确

## 技术栈

- **Streamlit**：Web应用框架
- **pandas**：数据处理
- **numpy**：数值计算
- **plotly**：交互式可视化
- **Pillow**：图像处理

## 作者

刘佳城 (2511210771)

## 数据来源

MIT GOSSIS Initiative - WiDS Datathon 2020
