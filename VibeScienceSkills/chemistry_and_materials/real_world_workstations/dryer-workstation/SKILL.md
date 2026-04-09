---
name: dryer-workstation
description: The electric thermostatic forced-air drying oven generates heat through electric heating elements and uses a fan to circulate the air inside the chamber, thereby drying materials.
---

### 烘干机
用于样品干燥、容器烘干、溶剂去除或测试前干燥处理
- **参数设置**：
- 操作：静置烘干
- 说明：用于对样品或容器进行控温干燥处理
- 参数：
   - 烘干时间：指定烘干时间（如`10`，单位为分钟，可选范围为1-14400分钟）
   - 烘干温度：指定烘干温度（如`30`，单位为摄氏度，可选范围为26-210摄氏度）
   - 容器类型：指定需烘干的容器类型（可选`进样瓶`或`50ml耐热瓶`）
   - 容器编号：指定需烘干的容器编号（如`[1,2]`，表示选择1号和2号容器，最大10个）

**参数约束**：
1. 恒温温度（℃）：26-210；
2. 烘干时间（分钟）：1-14400

#### 输入限制
1. 输入样品需处于可烘干状态；
2. 输入容器需与烘干环境兼容

#### 输出状态
1. 样品完成干燥处理；
2. 容器可进入后续转移、表征或存储流程