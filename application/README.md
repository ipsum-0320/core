# 算法记录

## 优化问题的分类
从约束函数个数的角度来分析，可分为两类：
* 无约束优化问题。
* 约束优化问题。


从解空间的分布情况来看，又可以被分为两类：
* 连续优化问题。
* 组合（即离散）优化问题。


优化问题有两类解决办法；
* 精算优化方法。
* 启发式算法。


常规的优化问题可以被描述为：`min f(x), g(x) >= 0, x \in D`，其中 f 是目标函数，x 是解，D 是定义域，g 是约束函数集。

## 生物地理学优化算法相关概念
生物地理学优化算法有如下关键概念：
* 适宜度指数 HSI，充当目标函数。
* 适宜度变量 SIVs，是一个向量，充当解。
* 迁入率 λ，迁出率 μ，变异率 π。

一个标准的 BBO 算法的步骤如下：
1. 随机生成问题的一组初始解。构成初始种群。
2. 计算种群中每个解的适宜度指数 HSI，并依此计算每个解的**迁入率**、**迁出率**和**变异率**。
3. 更新当前已经找到的最优解 Hb，如果 Hb 不在当前种群中，则将其加入到种群中。
4. 如果终止条件满足，返回当前已找到的最优解，算法结束，否则转至第 5 步。
5. 对于每个解进行迁移操作（算法有默认的迁移方式）。
6. 对于每个解进行变异操作（算法有默认的变异方式）。


## 生物地理学优化算法常见优化
1. 改进迁移操作，比如引入混合迁移，H(d) = aH(d) + (1 - a)H'(d)，这不仅达到了原始迁移操作所起到的信息交互作用，而且进一步避免了较优的解因迁移而产生质量下降的情况。
2. 修正迁移率，例如将迁入率和迁出率定义为二次模型或者三角函数模型，这能够更加真实的模拟自然界中栖息地的实际情况。
3. 修改种群的拓扑结构，将原本理想的全局拓扑结构优化为更符合实际的局部拓扑结构。


## 集大成的生态地理学优化算法
1. 全新的迁移模型（修改迁移操作），局部迁移和全局迁移。
2. 成熟度控制，局部迁移和全局迁移并不是同步进行的，算法将会使用成熟度来控制每次迁移操作究竟是使用局部迁移还是全局迁移。这里的成熟度是一个动态变化的参数。


## 生物地理学优化算法在多目标优化问题中的应用
1. 聚集函数来作为算法的选择机制，将时延和能耗的优化结合成一个目标优化函数（优化，需要确定多个权向量）。
2. 群体初始化策略进行了一定的改进。
3. 基于自适应的变异率。
4. 调整变异策略，增大结果突变为中心云的概率。
5. 挑选最优结果时选择按照聚集函数计算出来的 HSI 值进行排序。
6. s 和 HSI 的映射方式。
7. 精英保存策略，h = max(h, h')，或者存储最优解，后续拿最优解替换最差解。

## 自己的算法思路
1. 选择合适的群体初始化策略。
2. 选取合适的 HSI 定义方式。
3. 选取合适的寻优方式。
4. 选取合适的拓扑结构进行迁移（生态地理学优化算法）。
5. 选取合适的变异方式。