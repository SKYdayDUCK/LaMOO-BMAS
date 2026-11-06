# LaMOO-BMAS

[中文](README.md) | [English](README.en.md)

**Batched Multi-objective Adaptive Search · 批量多目标自适应搜索**

把已测参数及其指标交给搜索器，让它推荐下一批值得尝试的参数。反复执行“推荐 → 评估 → 回传”，逐步更新参数空间的性能模型和已测帕累托前沿。

LaMOO-BMAS 不依赖具体架构。你负责定义参数、运行实验和计算指标；搜索器负责选点、分配配置预算、保存状态和提取性能权衡。可以接入普通函数、批量矩阵、任务队列或远程实验服务。

Python 包名为 `lamoo-bmas`，导入名为 `pareto_search`。依赖 NumPy、SciPy 和 scikit-learn，搜索模型在 CPU 上运行。

## 安装

```sh
git clone https://github.com/SKYdayDUCK/LaMOO-BMAS.git
cd LaMOO-BMAS
python -m pip install .
```

也可以直接安装：

```sh
python -m pip install "git+https://github.com/SKYdayDUCK/LaMOO-BMAS.git"
```

## 输入和输出

|接口|输入|输出|
|---|---|---|
|`Optimizer(...)`|参数范围、指标名、尺度、预算、共同种子；可选历史与优先级|持久化搜索器|
|`ask(8)`|希望取得的候选数量|最多 8 套参数，每套含稳定 ID|
|外部评估|一套参数与一个种子|按固定顺序排列的指标向量|
|`tell(batch_id, rows)`|候选参数、逐种子指标、稳定批次 ID|保存观测，更新搜索状态|
|`frontier()`|当前已收集的观测|共同种子均值上的非支配配置及其指标|
|`representative()`|当前完整观测与可选优先级|一套参数的完整指标、档位和加权分数|
|`status()`|当前状态|可提出数量、预算占用、已观测配置数及结束原因|

例如搜索 `(learning_rate, window)`，优化 `(regret, latency)`。`ask` 返回的列表形如：

```python
[
    {"id": "candidate-a", "params": {"learning_rate": 0.03, "window": 64}},
    {"id": "candidate-b", "params": {"learning_rate": 0.08, "window": 128}},
]
```

每套参数的结果以如下格式回传：

```python
{
    "candidate": point,
    "objectives": ["regret", "latency"],
    "minimize": True,
    "per_seed": [
        {"seed": 42, "values": [21.5, 8.2]},
        {"seed": 123, "values": [22.1, 8.0]},
    ],
}
```

所有目标统一最小化。最大化指标可取负后输入，例如负准确率；变换规则和指标顺序在一次搜索中保持固定。`scale` 表示各指标的量级，让不同单位的指标进入可比较的搜索坐标。

## 最小接入代码

```python
from pareto_search import Optimizer

search = Optimizer(
    "results/search",
    space={
        "learning_rate": {"low": 0.001, "high": 1.0, "log": True},
        "window": {"low": 16, "high": 512, "integer": True},
    },
    objectives=["regret", "latency"],
    budget=64,
    batch_size=8,
    seeds=[42, 123],
    target=[0.0, 0.0],
    scale=[100.0, 50.0],
    context={"scenario": "v1"},
)

while points := search.ask(8):
    # evaluate 是你的评估函数，返回 [regret, latency]。
    # 此处也可以批量提交到已有矩阵执行器或任务队列。
    rows = [
        {
            "candidate": point,
            "objectives": search.objectives,
            "minimize": True,
            "per_seed": [
                {"seed": seed, "values": evaluate(point["params"], seed)}
                for seed in search.seeds
            ],
        }
        for point in points
    ]
    search.tell("batch:" + ":".join(p["id"] for p in points), rows)

print(search.frontier())
print(search.status())
```

`budget=64` 表示新增 64 套参数，`batch_size=8` 表示每次推荐最多 8 套，两个种子表示每套进行两个共同重复，共最多 128 次筛选运行。剩余额度或候选不足时自动缩小批次。`run_budget` 可以进一步限制“配置×种子”的运行总数。

参数空间是数值轴构成的直积空间，支持线性连续轴、对数连续轴和整数轴。对数轴使用正上下界，整数轴使用整数上下界。种子至少两个，逐种子数据用于估计观测波动。

## 指标优先级与联合代表

从0.2.0起，可以通过`priority`设置主指标、第二层指标和辅助指标。以下示例对五个目标使用70%、25%、5%的组权重：

```python
search = Optimizer(
    "results/priority",
    space={"learning_rate": {"low": 0.001, "high": 1.0, "log": True}},
    objectives=["regret", "latency", "worst_regret", "p95_latency", "switch_rate"],
    budget=64,
    seeds=[42, 123],
    scale=[100, 50, 150, 90, 1],
    priority={
        "levels": [[0, 1], [2, 3], [4]],
        "weights": [0.70, 0.25, 0.05],
        "reference": [100, 50, 150, 90, 0.4],
        "reducers": ["max", "max", "mean"],
        "required_levels": 2,
    },
)
# 按前面的ask/tell循环评价后：
print(search.representative())
```

`levels`按重要性排列，索引对应`objectives`，每项指标恰好属于一层。`weights`为非负组权重、总和为1；`reducers`支持`max`和`mean`，默认各层取`max`。`reference`是事先固定的逐指标参照值。`required_levels`指定参与达标分档的前几层，默认取前两层；设为0时直接按加权分数排序。

目标均越小越好，严格低于参照视为达标。上例先分为三档：前两层全部达标、仅第一层全部达标、第一层存在未达标；同档再按加权分数排序。设逐种子均值为`f[j]`，计算`r[j] = f[j] / max(abs(reference[j]), 1e-12)`，再取`v[j] = r[j] / (1 + abs(r[j]))`。分数为各层归约值的加权和；上例为`0.70*max(v[0:2]) + 0.25*max(v[2:4]) + 0.05*v[4]`。这一变换也保留负目标的大小顺序。

优先级同时参与搜索：联合排名生成分区标签，前几层的联合非支配配置用于区域内的前沿邻近筛选，局部与全局LCB名额依据预测的档位和联合分数选点；保留距离探索名额。`frontier()`始终返回全部目标上的已测非支配集合。`representative()`返回一套完整配置及其所有指标，另带`tier`、`score`、`levels_passed`和`contributions`；无完整观测时返回`None`。

优先级写入冻结协议，恢复时使用相同设置。未配置`priority`时沿用原有目标组轮换与选点；此时`representative()`按`target`、`scale`归一化后的最大目标值选取一套配置。

## 两个可运行使用场景

### 场景一：在线策略的遗憾与切换率

文件：[examples/bandit_tradeoff.py](examples/bandit_tradeoff.py)

模拟 4 个选项、400 个时间步；每 100 步更换表现最好的选项。搜索学习率和探索比例，评估累计动态伪遗憾与动作切换率。较快追踪变化和减少切换构成两个需要共同观察的目标。

```sh
python examples/bandit_tradeoff.py --output results/bandit --budget 16
```

环境和动作采样使用分开的随机流，各参数配置使用相同种子。将 `evaluate(params, seed)` 替换为自己的环境，即可复用同一循环。

### 场景二：分类模型的验证错误率与模型规模

文件：[examples/model_tradeoff.py](examples/model_tradeoff.py)

使用 scikit-learn 自带 Wine 数据集。搜索随机森林的树数量和最大深度，评估验证错误率与训练后所有树的节点总数，展示预测效果和模型规模之间的折中。

```sh
python examples/model_tradeoff.py --output results/model --budget 16
```

每个种子固定训练/验证划分。搜索读取验证指标；最终参数选定后，可以另行在独立测试集评价。

两个示例生成 `frontier.json` 与 `observations.json`。前者展示当前性能权衡，后者可作为历史初始化输入。另有 [examples/minimal.py](examples/minimal.py)，用两个简单函数展示基本接口。

## 冷启动、历史初始化和恢复

默认冷启动，首先覆盖参数范围；收到完整观测后逐步建立模型。已有数据可按 `tell` 的行格式显式传入构造器的 `history=rows` 参数。例如继续第一个示例：

```sh
python examples/bandit_tradeoff.py --output results/bandit-next --budget 16 --history results/bandit/observations.json
```

- 历史参数、目标、场景和种子应与当前实验匹配。包含全部筛选种子的配置用于训练；额外种子数据保留。历史配置单独记录，本轮预算计入新增配置。
- 相同输出目录和相同配置可恢复搜索。初始化时传入的历史列表，恢复时也传入同一份。
- 参数顺序、目标、尺度、预算、场景版本及初始化数据哈希写入冻结协议。新协议使用新目录。
- `ask` 保存预留候选；未收齐共同种子时，重复调用返回同一批。外部执行器可依据任务回执复用已完成评估。
- `tell` 按批次 ID 幂等提交；校验失败或写盘失败时回滚本次内存修改。
- 每个目录由一个控制进程管理。多个评估 worker 可并行工作，由控制进程汇总后调用 `tell`。

## 内部如何决定下一批？

```mermaid
flowchart LR
    A[参数范围和优化目标] --> B[冷启动或历史观测]
    B --> C[归一化与逐指标代理模型]
    C --> D[空间分区与区域选择]
    D --> E[批量候选推荐]
    E --> F[外部实验评估]
    F --> G[逐种子指标回传]
    G --> C
    G --> H[更新已测帕累托前沿]
    H --> D
```

1. **统一坐标。** 参数映射到归一化空间，对数轴先取对数；目标按预设 `target` 和 `scale` 转换。
2. **学习性能和不确定性。** 完整观测达到 4 套时，为每项指标拟合一个高斯过程。共同种子的波动参与观测噪声估计。
3. **寻找有希望的区域。** 根据观测的支配计数形成标签，用 RBF SVM 划分空间；数据足够时递归细分，结合采样超体积和探索项选择区域。区域内候选还结合已测前沿附近的位置筛选，使前沿参与下一批选点。启用优先级时，分区标签改由联合排名生成。
4. **形成候选池。** 在参数范围内生成 Sobol 候选，去掉已经观测或预留的等效点。
5. **兼顾局部改进与全局探索。** 完整 8 点批次中，最多 4 点来自区域内的 LCB 选点，2 点考虑全目标归一化差距，2 点探索距已知配置较远的位置。点间距离惩罚用于分散本批候选。启用优先级时，前6个名额按预测档位与加权分数选点。
6. **更新前沿。** 所有配置按指定种子的均值比较。如果某配置在各指标上都不差，并至少一项更好，就支配另一配置；保留未被支配的配置。

未启用优先级时，指标较多则按组轮换，每组最多三项，同时使用全目标选点路线和全目标前沿。可配置 `groups=[[0,1,2],[1,3]]` 等覆盖全部指标的组别。超体积采用固定采样盒估计，盒子由首次完整观测确定。

例如 A 的 `(遗憾, 延迟)` 为 `(20,10)`，B 为 `(25,8)`，两者各有优势，都会留在前沿；C 为 `(28,12)`，被 A 支配。下一批会继续尝试扩展这些性能权衡。前沿由实际评估点构成，预测用于选择评估位置。

## 多算法与停止规则

不同算法各创建一个 `Optimizer`，独立定义参数、预算和状态。调用方将各自的 `ask` 结果统一执行，再分算法 `tell`。`method="grid"` 提供单轴网格补点，`method="fixed"` 用于固定配置。

库默认在配置预算、运行预算或有效候选用尽时结束，`ask` 返回空列表；调用方也可在循环外按自己的验证标准停止。统计复核、业务目标判定和最终测试协议由外部实验系统管理。

## 方法来源与工程实现

LaMOO-BMAS 借鉴 LaMOO 的支配关系分区、帕累托引导和区域探索思想，独立实现全局高斯过程、LCB 批量选点及目标组轮换。历史初始化、预算预留、尾批缩容和状态恢复用于对接外部实验。

- [LaMOO 原仓库](https://github.com/aoiang/LaMOO)
- [Multi-objective Optimization by Learning Space Partition — ICLR 2022](https://openreview.net/forum?id=FlwzVjfMryn)
- [Multi-Objective Neural Architecture Search by Learning Search Space Partitions — JMLR 2024](https://www.jmlr.org/papers/v25/23-1013.html)

## 测试与许可证

```sh
python -m pip install pytest
python -m pytest tests -q
```

测试覆盖冷启动、历史初始化、不同目标维数、参数类型、种子完整性、批次幂等、写盘失败回滚、恢复、预算缩容与可选优先级选择。

本项目采用 [MIT License](LICENSE)。
