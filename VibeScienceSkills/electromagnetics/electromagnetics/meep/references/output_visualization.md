# Output and Visualization in Meep

輸出和可視化用於分析和展示仿真結果。

## HDF5輸出

### 基本HDF5輸出

```python
import meep as mp

# 創建仿真
sim = mp.Simulation(...)

# 輸出介電常數
sim.run(mp.at_beginning(mp.output_epsilon))

# 輸出電場
sim.run(mp.output_efield_z)  # z分量
sim.run(mp.output_hfield_x)  # x分量
```

### 時序輸出

```python
# 在每個時間步輸出
sim.run(mp.at_every(0.6, mp.output_efield_z))

# 在特定時間輸出
sim.run(mp.at_time(100, mp.output_efield_z))
```

### 追加輸出

```python
# 追加輸出到單個HDF5文件
sim.run(mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)))
```

這會創建一個三維HDF5文件，第三維是時間。

### 輸出所有場分量

```python
# 輸出所有電場分量
sim.run(mp.output_efield)

# 輸出所有磁場分量
sim.run(mp.output_hfield)

# 輸出所有場分量
sim.run(mp.output_total_poynting)
```

## PNG輸出

### 基本PNG輸出

```python
# 輸出PNG圖像
sim.run(mp.output_png(mp.Ez, "-Zc dkbluered"))
```

### PNG選項

```python
# 顏色映射選項
sim.run(mp.output_png(mp.Ez, "-Zc dkbluered"))  # 藍藍色映射
sim.run(mp.output_png(mp.Ez, "-Zc rdgyb"))    # 紅綠色映射

# 其他選項
sim.run(mp.output_png(mp.Ez, "-R"))        # 一致顏色範圍
sim.run(mp.output_png(mp.Ez, "-S y"))      # y軸翻轉
```

### 時序PNG輸出

```python
# 在每個時間步輸出PNG
sim.run(mp.at_every(0.6, mp.output_png(mp.Ez, "-Zc dkbluered")))
```

## 自定義輸出函數

### 基本自定義輸出

```python
def custom_output(sim):
    # 獲取場數據
    ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    
    # 處理數據
    # ... 處理邏輯
    
    # 保存到文件
    # ...

sim.run(mp.at_every(0.6, custom_output))
```

### 場取場數據

```python
# 獲取單點的場值
pt = mp.Vector3(0, 0, 0)
ez_value = sim.get_field_point(mp.Ez, pt)

# 獲取陣列場數據
ez_array = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# 獕取特定體積的場數據
volume = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(5, 5, 0))
ez_array = sim.get_array(component=mp.Ez, where=volume)
```

### 獲取介電常數

```python
# 獕取介電常數分佈
eps_array = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

# 獕取特定頻率的介電常數
freq = 0.15
eps_array = sim.get_epsilon_grid(frequency=freq)
```

## 輸出目錄

### 設定輸出目錄

```python
# 設定輸出目錄
sim.use_output_directory("my_simulation")
```

所有輸出文件將被放入`my_simulation/`目錄。

### 輸出文件名前綴

```python
# 設定文件名前綴
sim = mp.Simulation(..., filename_prefix="my_sim_")
```

### 獲取文件名前綴

```python
# 獲取當前文件名前綴
prefix = sim.get_filename_prefix()
print(f"Output prefix: {prefix}")
```

## 場積輸出

### NumPy可視化

```python
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# 創建和運行仿真
sim = mp.Simulation(...)
sim.run(until=200)

# 獲取場數據
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

# 繪製圖像
plt.figure()
plt.imshow(eps_data.transpose(), cmap='binary')
plt.imshow(ez_data.transpose(), cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.colorbar()
plt.show()
```

### 多幀圖顯示

```python
# 獲取多個場分量
ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)
ey_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# 繢製多幀圖
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ex_data.transpose(), cmap='RdBu')
axes[0].set_title('Ex')
axes[0].axis('off')

axes[1].imshow(ey_data.transpose(), cmap='RdBu')
axes[1].set_title('Ey')
axes[1].axis('off')

axes[2].imshow(ez_data.transpose(), cmap='RdBu')
axes[2].set_title('Ez')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 時序可視化

```python
# 獲取時序場數據
ez_times = []

def collect_ez(sim):
    ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    ez_times.append(ez.copy())

sim.run(mp.at_every(0.6, collect_ez))

# 繰製動畫
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
im = ax.imshow(ez_times[0].transpose(), cmap='RdBu', animated=True)
ax.axis('off')

def update(frame):
    im.set_array(ez_times[frame].transpose())
    return [im]

ani = FuncAnimation(fig, update, frames=len(ez_times), interval=50)
plt.show()
```

## 體積輸出

### 使用h5utils

```python
# 使用h5topng轉換HDF5到PNG
import subprocess

# 轉換單個時間步
subprocess.run(["h5topng", "-t", "100", "ez.h5"])

# 轉換所有時間步
subprocess.run(["h5topng", "-t", "0:332", "ez.h5"])

# 帶用顏色映射
subprocess.run(["h5topng", "-Zc", "dkbluered", "ez.h5"])

# 疺加介電常數背景
subprocess.run(["h5topng", "-A", "eps-000000.00.h5", "ez.h5"])
```

### 創建動畫

```python
# 獕換所有時間步到PNG
subprocess.run(["h5topng", "-t", "0:332", "-R", "-Zc", "dkbluered", "ez.h5"])

# 轉換為動畫
subprocess.run(["convert", "ez.t*.png", "ez.gif"])
```

## 場積輸出

### 體積分析

```python
# 獲取電場能量
electric_energy = sim.electric_energy_in_box(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(10, 10, 0)
)

# 獲取磁場能量
magnetic_energy = sim.magnetic_energy_in_box(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(10, 10, 0)
)

# 獕取總能量
total_energy = sim.field_energy_in_box(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(10, 10, 0)
)
```

### 能量密度

```python
# 獲取電場能量密度
def electric_energy_density(sim):
    Ex = sim.get_field_point(mp.Ex, pt)
    Ey = sim.get_field_point(mp.Ey, pt)
    Ez = sim.get_field_point(mp.Ez, pt)
    return 0.5 * (abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2)

# 獲取磁場能量密度
def magnetic_energy_density(sim):
    Hx = sim.get_field_point(mp.Hx, pt)
    Hy = sim.get_field_point(mp.Hy, pt)
    Hz = sim.get_field_point(mp.Hz, pt)
    return 0.5 * (abs(Hx)**2 + abs(Hy)**2 + abs(Hz)**2)
```

### 坡印廷向量

```python
# 獲取場分量
def poynting_vector(sim):
    Ex = sim.get_field_point(mp.Ex, pt)
    Ey = sim.get_field_point(mp.Ey, pt)
    Ez = sim.get_field_point(mp.Ez, pt)
    Hx = sim.get_field_point(mp.Hx, pt)
    Hy = sim.get_field_point(mp.Hy, pt)
    Hz = sim.get_field_point(mp.Hz, pt)
    
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    
    return mp.Vector3(Sx, Sy, Sz)
```

## 頻譜輸出

### 頻譜可視化

```python
# 獲取頻譜數據
harminv = mp.Harminv(component=mp.Ez,
                       frequency=0.15,
                       decay_by=0.001)

# 添加到仿真
sim = mp.Simulation(...)
sim.run(harminv)

# 獲取頻譜
modes = harminv.modes
for mode in modes:
    print(f"Frequency: {mode.freq}")
    print(f"Decay rate: {mode.decay}")
    print(f"Q factor: {mode.freq / (2*mode.decay)}")
```

### 頻譜場可視化

```python
# 獲取頻譜場模式
for mode in modes:
    # 獲取頻譜場
    mode_field = mode.get_field(sim)
    
    # 繢製頻譜場
    plt.figure()
    plt.imshow(mode_field.transpose(), cmap='RdBu')
    plt.title(f"Mode at {mode.freq:.4f}")
    plt.axis('off')
    plt.show()
```

## 輸出優化

### 減小輸出文件大小

```python
# 只輸出需要的場分量
sim.run(mp.output_efield_z)  # 只輸出Ez

# 使用PNG代替HDF5
sim.run(mp.output_png(mp.Ez, "-Zc dkbluered"))

# 輸出特定體積
sim.run(mp.in_volume(mp.Volume(center=mp.Vector3(0, 0, 0), 
                           size=mp.Vector3(5, 5, 0)),
            mp.output_efield_z))
```

### 減少輸出頻率

```python
# 降低輸出頻率
sim.run(mp.at_every(1.2, mp.output_efield_z))  # 從0.6改為1.2

# 使用時序輸出
sim.run(mp.at_time(100, mp.output_efield_z))  # 只在特定時間輸出
```

### 批量輸出

```python
# 使用輸出目錄
sim.use_output_directory("my_simulation")

# 使用文件名前綴
sim = mp.Simulation(..., filename_prefix="my_sim_")
```

## 輸出故障排除

### 輸出文件太大

**原因：**
1. 輸出頻率太高
2. 輸出體積太大
3. 使用HDF5代替PNG

**解決方法：**
1. 降低輸出頻率
2. 輸出特定體積
3. 使用PNG輸出

### 輸出文件不存在

**原因：**
1. 輸出函數沒有被調用
2. 輸出目錄不正確
3. 文件權限問題

**解決方法：**
1. 確認輸出函數被調用
2. 檢查輸出目錄
3. 檢查文件權限

### 可視化不正確

**原因：**
1. 場取數據不正確
2. 數據不正確
3. 顏色映射不正確

**解決方法：**
1. 檢查數據獲取
2. 確認數據維度
3. 誢查顏色映射

### 動畫不流暢

**原因：**
1. 數據太大
2. 輸出頻率太高
3. 系統資源不足

**解決方法：**
1. 減小圖像尺寸
2. 降低幀率
3. 減少幀圖數量

## 高級輸出

### 場積輸出

```python
# 輸出場能量時序
energies = []

def collect_energy(sim):
    energy = sim.field_energy_in_box(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(10, 10, 0)
    )
    energies.append(energy)

sim.run(mp.at_every(0.6, collect_energy))

# 繃製能量時序
plt.figure()
plt.plot(energies)
plt.xlabel('Time step')
plt.ylabel('Energy')
plt.title('Total field energy')
plt.show()
```

### 場積輸出

```python
# 輸出多個位置的場時序
positions = [mp.Vector3(-5, 0, 0), mp.Vector3(0, 0, 0), mp.Vector3(5, 0, 0)]
field_values = {pos: [] for pos in positions}

def collect_fields(sim):
    for pos in positions:
        ez = sim.get_field_point(mp.Ez, pos)
        field_values[pos].append(ez)

sim.run(mp.at_every(0.6, collect_fields))

# 繺製場時序
plt.figure()
for pos, values in field_values.items():
    plt.plot(values, label=f"x={pos.x}")
plt.xlabel('Time step')
plt.ylabel('Ez field')
plt.legend()
plt.show()
```

### 三維可視化

```python
# 使用Mayavi進行三維可視化
from mayavi import mlab

# 獲取三維場數據
ez_3d = sim.get_array(center=mp.Vector3(), size=cell_3d, component=mp.Ez)

# 繃製三維圖像
mlab.contour3d(ez_3d)
```

## 輸出最佳實踐

### 輸出策略

1. **快速測試**：使用PNG輸出
2. **數據分析**：使用HDF5.輸出
3. **可視化**：使用NumPy/Matplotlib
4. **動畫**：使用時序輸出

### 輸出頻率

1. **快速**：每1-2個時間步
2. **正常**：每0.5-1個時間步
3. **高精度**：每0.1-0.5個時間步

### 輸出體積

1. **全網格**：輸出整個計算域
2. **感興趣區域**：輸出特定體積
3. **截面**：輸出二維截面

### 輸出格式

1. **PNG**：快速可視化
2. **HDF5**：數據分析
3. **二進制**：高精度數據