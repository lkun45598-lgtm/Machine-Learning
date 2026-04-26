# CLAUDE.md

## 环境
- conda 环境：`pytorch`
- 运行方式：`conda run -n pytorch python xxx.py`

## 项目结构
- 按周分文件夹：`week1/`、`week2/`、...
- 每周包含：实验代码、数据集、`README.md`
- 根目录 `README.md` 维护各周实验汇总表格

## 代码规范
- 缺失值填充必须在 `train_test_split` 之后，只用训练集的统计量填充，避免数据泄漏
- `StandardScaler` 同理，只对训练集 `fit`，测试集只 `transform`
- 数据预处理顺序：切分 → 填充缺失值 → 标准化

## GitHub
- 仓库：`github.com/lkun45598-lgtm/Machine-Learning`
- 不上传 `.docx` 文件
- 每周实验完成后推送，commit message 格式：`Week X: 实验内容简述`
