Automatic Distributed Training
=====================

[//]: # (- 支持多种分布式并行策略)
[//]: # (	- 数据并行)
[//]: # (	-  模型并行)
[//]: # (	-  流水线并行)
[//]: # (- 支持常见分布式通信架构)
[//]: # (	- 参数服务器)
[//]: # (	- 集合通信)
[//]: # (- 支持硬件感知的自动化分布式训练)
[//]: # (	- 计算图切割)
[//]: # (	- 代价估计)
[//]: # (	- 动态规划)

- Supporting mulitple parallel training schemas
	- Data parallelism
	- Model parallelism
		- Parameter parallelism
		- Operator parallelism
		- Pipeline parallelism
		- Subgraph parallelism
		- Optimizer parallelism
- Supporting popular distributed communication architectures
	- Parameter server
	- Collective communication
- Supporting topology-aware automatic distributed training
	- Device placement
	- Cost estimation
	- Dynamic programming


![](distributed.png)