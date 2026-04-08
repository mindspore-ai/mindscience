# Flux Analysis in Meep

通量分析用於計算透射、反射和散射譜。

## 基本通量計算

### 添加通量監控器

```python
import meep as mp

# 創建仿真
sim = mp.Simulation(...)

# 定義通量參數
fcen = 0.15  # 中心頻率
df = 0.1     # 頻率寬度
nfreq = 100  # 頻率點數

# 添加通量監控器
flux_region = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2))
flux = sim.add_flux(fcen, df, nfreq, flux_region)
```

### 獲取通量譜

```python
# 運行仿真
sim.run(until=200)

# 獕取通量譜
flux_spectrum = mp.get_fluxes(flux)  # 通量譜
freqs = mp.get_flux_freqs(flux)  # 頻率列表
```

### 顯示通量譜

```python
# 顯示通量譜
mp.display_fluxes(flux)
```

## 透射/反射譜

### 單次運行方法

對於準確的透射/反射計算，需要兩次運行：

1. **規範化運行**：空網格或參考結構
2. **實際運行**：包含散射體的結構

### 規次運行方法

```python
import meep as mp
import numpy as np

# 定義參數
fcen = 0.15
df = 0.1
nfreq = 100

# ========== 第一次運行：規範化 ==========
# 設建參考結構（直波導）
geometry = [mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=12))]

# 創建仿真
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# 添加通量監控器
refl_fr = mp.FluxRegion(center=mp.Vector3(-5, 0), size=mp.Vector3(0, 2))
tran_fr = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2))

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# 運行直到場衰減
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(5, 0), 1e-3))

# 保存規範化通量數據
incident_flux = mp.get_fluxes(tran)
refl_data = sim.get_flux_data(refl)

# ========== 第二次運行：實際結構 ==========
# 重置仿真
sim.reset_meep()

# 創建實際結構（彎波導）
geometry = [
    mp.Block(size=mp.Vector3(12, 1, mp.inf),
             center=mp.Vector3(-2.5, -3.5, 0),
             material=mp.Medium(epsilon=12)),
    mp.Block(size=mp.Vector3(1, 12, mp.inf),
             center=mp.Vector3(3.5, 2, 0),
             material=mp.Medium(epsilon=12))
]

# 創建新仿真
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# 重新添加通量監控器
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran_fr = mp.FluxRegion(center=mp.Vector3(3.5, 2), size=mp.Vector3(2, 0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# 減入規範化反射數據（負號）
sim.load_minus_flux_data(refl, refl_data)

# 運行仿真
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(3.5, 2), 1e-3))

# 獲取實際通量
bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)
```

### 計算透射/反射

```python
# 計算透射譜
transmittance = bend_tran_flux / incident_flux

# 計算反射譜
reflectance = -bend_refl_flux / incident_flux

# 計算散射損耗
loss = 1 - transmittance - reflectance
```

### 繢製譜

```python
import matplotlib.pyplot as plt

# 獲取頻率
freqs = mp.get_flux_freqs(tran)
wavelengths = 1.0 / freqs

# 繢製譜
plt.figure()
plt.plot(wavelengths, transmittance, 'r-', label='Transmittance')
plt.plot(wavelengths, reflectance, 'b-', label='Reflectance')
plt.plot(wavelengths, loss, 'g-', label='Loss')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Power')
plt.legend()
plt.show()
```

## 通量監控器配置

### 通量區域

```python
# 點通量區域（線）
flux_region = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2))

# 面通量區域（面）
flux_region = mp.FluxRegion(center=mp.Vector3(0, 0, 0), size=mp.Vector3(10, 10, 0))

# 體通量區域（體）
flux_region = mp.FluxRegion(center=mp.Vector3(0, 0, 0), size=mp.Vector3(10, 10, 10))
```

### 通量方向

```python
# 指定通量方向
flux_region = mp.FluxRegion(center=mp.Vector3(5, 0), 
                         size=mp.Vector3(0, 2, 0),
                         direction=mp.X)  # x方向通量
```

### 多個通量監控器

```python
# 多個通量監控器
flux1 = sim.add_flux(fcen, df, nfreq, flux_region1)
flux2 = sim.add_flux(fcen, df, nfreq, flux_region2)
flux3 = sim.add_flux(fcen, df, nfreq, flux_region3)

# 顯示所有通量
mp.display_fluxes(flux1, flux2, flux3)
```

## 散射通量

### 散射截面

```python
# 包圍散射體的封閉通量盒
r = 1.0  # 散射體半徑

# 六個面的通量盒
box_x1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector.3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector.3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector.3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector.3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector.3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector.3(2*r, 2*r, 0)))
```

### 散射功率

```python
# 獲取所有面的通量
total_flux = (sum(mp.get_fluxes(box_x1)) + sum(mp.get_fluxes(box_x2)) + \
              (sum(mp.get_fluxes(box_y1)) + sum(mp.get_fluxes(box_y2)) + \
              (sum(mp.get_fluxes(box_z1)) + sum(mp.get_fluxes(box_z2)))
```

### 散射截面

```python
# 獲取入射功率（從規範化運行）
incident_power = sum(incident_flux)

# 獲取散射功率（從實際運行）
scattered_power = total_flux - incident_power

# 獕取散射截面
scattering_cross_section = scattered_power / incident_intensity
```

## 能量通量

### 添加能量監控器

```python
# 添加能量監控器
energy_region = mp.EnergyRegion(center=mp.Vector3(0, 0, 0), 
                           size=mp.Vector3(10, 10, 10))
energy = sim.add_energy(fcen, df, nfreq, energy_region)
```

### 獕取能量譜

```python
# 獲取能量譜
energy_spectrum = mp.get_energy(energy)
freqs = mp.get_flux_freqs(energy)
```

## 力通量

### 添加力監控器

```python
# 添加力監控器
force_region = mp.ForceRegion(center=mp.Vector3(0, 0, 0), 
                         size=mp.Vector3(10, 10, 10))
force = sim.add_force(fcen, df, nfreq, force_region)
```

### 獕取力譜

```python
# 獕取力譜
force_spectrum = mp.get_forces(force)
freqs = mp.get_flux_freqs(force)
```

## 通量數據管理

### 保存通量數據

```python
# 保存通量數據到HDF5文件
mp.save_flux("refl_data", refl)
```

### 載入通量數據

```python
# 從HDF5文件載入通量數據
mp.load_flux("refl_data", refl)
```

### 減入負通量數據

```python
# 減入負通量數據（用於反射計算）
sim.load_minus_flux_data(refl, refl_data)
```

## 通量與場衰減

### 場止條件

```python
# 使用場衰減作為停止條件
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50,              # 檢查時間間隔
    mp.Ez,           # 場查的場分量
    mp.Vector3(5, 0), # 繢查位置
    1e-3              # 衰減閾值
))
```

### 手動停止條件

```python
# 手動定義停止條件
def stop_condition(sim):
    # 檢查場是否衰減到足夠小
    ez = sim.get_field_point(mp.Ez, mp.Vector3(5, 0))
    return abs(ez) < 1e-3

sim.run(until_after_sources=stop_condition)
```

## 通量與解析度

### 解析度要求

```python
# 基本規則：10-20像素/波長
wavelength = 1.0 / fcen
resolution = 15 / wavelength  # 15像素/波長

sim = mp.Simulation(..., resolution=resolution)
```

### 收斂測試

```python
# 解析度收斷測試
resolutions = [10, 15, 20]
results = []

for res in resolutions:
    sim = mp.Simulation(..., resolution=res)
    # ... 運行仿真
    results.append(transmittance)

# 比較結果
print(f"Resolution 10: {results[0]}")
print(f"Resolution 15: {results[1]}")
print(f"Resolution 20: {results[2]}")
```

## 通量與頻帶材料

### 頻帶材料通量

```python
# 頻帶材料的通量計算
susceptibility = mp.LorentzianSusceptibility(frequency=0.3, gamma=0.1, sigma=0.5)
material = mp.Medium(epsilon=3.4, susc=susceptibility)

geometry = [mp.Block(material=material, ...)]
```

### 增益材料通量

```python
# 增益材料的通量計算
material = mp.Medium(epsilon=3.4, D_conductivity=-0.1)

geometry = [mp.Block(material=material, ...)]
```

## 最佳實踐

### 通量監控器位置

```python
# 通量監控器應該在PML外部
cell = mp.Vector3(16, 8, 0)
pml_layers = [mp.PML(1.0)]

# 通量監控器在PML(部（正確）
flux_fr = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2))

# 通量監控器在PML內部（錯誤）
flux_fr = mp.FluxRegion(center=mp.Vector3(7.5, 0), size=mp.Vector3(0, 2))
```

### 通量監控器尺寸

```python
# 通量監控器應該覆蓋整個模式
waveguide_width = 1.0

# 通量監控器尺寸應該大於波導寬度
flux_size = 2.0 * waveguide_width
flux_fr = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, flux_size))
```

### 頻率範圍選擇

```python
# 頻率範圍應該匹配源頻譜範圍
fcen = 0.15  # 中心頻率
df = 0.1     # 頻率寬度

# 頻率範圍：[fcen - df/2, fcen + df/2]
# 確保覆蓋感興趣的頻率範圍
```

### 頻率點數選擇

```python
# 頻率點數影響解析度和計算量
nfreq = 50   # 少點數：快速測試
nfreq = 100  # 中等點數：平衡
nfreq = 200  # 多點數：高精度
```

## 故障排除

### 通量譜不正確

**原因：**
1. 通量監控器位置不正確
2. 通量監控器尺寸不夠
3. 規範化運行不正確

**解決方法：**
1. 檢查通量監控器位置
2. 增加通量監控器尺寸
3. 確認規範化運行正確

### 透射/反射不守恆

**原因：**
1. 場次運行設置不一致
2. 通量監控器位置不同
3. 場行時間不足

**解決方法：**
1. 確認兩次運行設置一致
2. 使用相同的通量監控器位置
3. 增加運行時間

### 場率解析度低

**原因：**
1. 頻率點數太少
2. 頻率寬度太大
3. 運行時間不足

**解決方法：**
1. 增加頻率點數
2. 減小頻率寬度
3. 增加運行時間

### 通量不收斂

**原因：**
1. 解析度不足
2. 計格大小不足
3. PML反射

**解決方法：**
1. 增加解析度
2. 增加網格大小
3. 增加PML厚度