---
name: centrifuge-workstation
description: The workstation is used for automated centrifugal separation of liquid samples, separating solid-liquid systems, and achieving sample purification or precipitation collection.
---

### 离心机
用于液体样品澄清、固液分离、沉淀收集和简单离心处理

- **参数设置**：
- 操作：离心
- 说明：对液体样品进行自动化离心分离
- 参数：
    - 位置：指定离心管摆放位置（如`[1,6]`，注意离心管需对称摆放）
    - 转速：指定离心机运转速度（如`3000`转每秒）
    - 离心时间（分）：指定离心进行的时长（如`5`分钟）

**参数约束**：
1. 位置：1-10；
2. 转速：1000-8000；
3. 离心时间（分）：1-99

#### 输入限制
1. 容器必须有盖子；
2. 容器总数必须为1-10之间的偶数；
3. 容器内液体体积不能超过30ml

#### 输出状态
- 样品完成离心分离
- 可进入上清液转移、纯化或后续处理流程