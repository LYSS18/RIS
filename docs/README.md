

## 项目结构

```
├── config/                     # 配置文件
│   └── settings.py            # 系统参数和常量
├── src/                       # 源代码模块
│   ├── core/                  # 核心系统组件
│   │   ├── network_deployment.py    # 基站和RIS部署
│   │   ├── network_selector.py      # 网络元素选择
│   │   └── ris_controller.py        # RIS切换和SINR计算
│   ├── models/                # 机器学习模型
│   │   ├── trajectory_predictor.py  # 原始LSTM轨迹预测
│   │   ├── improved_trajectory_predictor.py # 改进LSTM轨迹预测
│   │   ├── improved_trajectory_tester.py    # 改进模型测试
│   │   ├── original_trajectory_tester.py    # 原始模型测试
│   │   └── interference_classifier.py       # CNN干扰分类
│   ├── analysis/              # 性能分析
│   │   ├── performance_analyzer.py  # 系统性能评估
│   │   └── sinr_error_analyzer.py   # SINR误差分析
│   └── visualization/         # 绘图和可视化
│       ├── plotter.py         # 结果可视化
│       └── sinr_error_plotter.py    # 误差分析可视化
├── utils/                     # 工具函数
│   ├── distance_calculator.py # 距离和角度计算
│   └── data_generator.py      # 数据生成工具
├── models/                    # 训练模型存储
│   ├── traj_model_120.h5     # 原始轨迹预测模型
│   ├── improved_traj_model.keras # 改进轨迹预测模型
│   ├── traj_model_trueNorm.npy    # 归一化参数
│   └── cnn_ris_model.h5       # CNN干扰分类模型
├── results/                   # 分析结果和图表
│   ├── power_analysis.png     # 功率分析图表
│   ├── element_analysis.png   # 元素分析图表
│   ├── power_error_analysis.png   # 功率误差分析图表
│   ├── element_error_analysis.png # 元素误差分析图表
│   └── *.json                 # 数值结果文件
├── test_exp/                  # 实验验证
│   ├── health_risk_analysis.py    # 健康风险分析
│   └── prediction_error_analysis.png # 误差分析图表
├── 最新轨迹预测/               # 最新轨迹预测模型
│   ├── lstm.py                # 训练脚本
│   ├── try.py                 # 测试脚本
│   └── model.keras            # 最新模型
├── docs/                      # 文档
│   └── README.md              # 项目说明文档
├── test/                      # 测试文件（原始轨迹预测模型）
│   ├── lstm.py                # LSTM测试
│   ├── model.keras            # 测试模型
│   └── try.py                 # 测试脚本
├── Geolife Trajectories 1.3/  # 轨迹数据集
├── Processed/                 # 处理后的数据
└── main.py                    # 主执行脚本
```


## 使用

```bash
python main.py
```

### 菜单选项
1. **训练模型** - 训练LSTM和CNN模型
2. **运行功率分析** - 分析不同功率级别下的性能
3. **运行元素分析** - 分析不同智能反射面元素数量下的性能
4. **可视化网络部署** - 显示网络拓扑
5. **运行信噪比误差分析** - 分析轨迹预测误差
6. **运行所有分析** - 完整分析套件
7. **退出**

### 使用独立模块

#### 轨迹预测
```python
from src.models.trajectory_predictor import TrajectoryPredictor

predictor = TrajectoryPredictor()
predictor.train()  # 训练模型
predicted_trajectory = predictor.predict_trajectory(previous_points, steps=10)
```

#### 智能反射面控制
```python
from src.core.ris_controller import RISController

controller = RISController()
sinr = controller.calculate_ris_switch(
    user_coords, base_station_coords, ris_coords,
    interfering_users, always_on=0, sinr=1
)
```

#### 性能分析
```python
from src.analysis.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
power_results = analyzer.analyze_power_levels()
element_results = analyzer.analyze_element_counts()
```

## 配置

系统参数可在 `config/settings.py` 中修改：

- **物理参数**: 频率、波长、路径损耗
- **功率设置**: 发射功率、噪声功率
- **智能反射面参数**: 元素数量、部署半径
- **机器学习参数**: LSTM单元、CNN架构、训练轮数
- **仿真参数**: 用户数量、轨迹点数



## 依赖项

- TensorFlow/Keras (机器学习模型)
- NumPy (数值计算)
- Pandas (数据处理)
- Matplotlib (可视化)
- Python 3.7+

## 模型文件

训练好的模型保存在 `models/` 目录中：
- `traj_model_120.h5` - LSTM轨迹预测模型
- `cnn_ris_model.h5` - CNN干扰分类模型
- `traj_model_trueNorm.npy` - 归一化参数

## 结果

分析结果和图表保存在 `results/` 目录中：
- 性能对比图表
- 网络部署可视化
- 包含数值结果的JSON文件


