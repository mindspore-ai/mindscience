# Boundary Conditions in Meep

邊界條件定義了仿真計算域邊界的電磁場行為。

## 完全匹配層（PML）

### 基本PML

```python
# 四周PML（厚度1.0）
pml_layers = [mp.PML(1.0)]

# 創用PML
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    ...)
```

### 指定方向PML

```python
# 只在x方向設置PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.X)]

# 只在y方向設置PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.Y)]

# 只在z方向設置PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.Z)]
```

### 指定方向和側邊PML

```python
# 只在x正方向設置PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.X, side=mp.High)]

# 只在x負方向設置PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.X, side=mp.Low)]
```

### 多個PML層

```python
# 多個PML層
pml_layers = [
    mp.PML(thickness=1.0, direction=mp.X),
    mp.PML(thickness=1.0, direction=mp.Y),
    mp.PML(thickness=1.0, direction=mp.Z)
]
```

### PML參數

```python
# 完整PML參數
pml_layers = [mp.PML(
    thickness=1.0,      # PML厚度
    direction=mp.X,     # 方向
    side=mp.High,        # 側邊
    pml_type=mp.PML,     # PML類型
    R_asymptote=True,     # R對稱性
    K_asymptote=False,    # K對稱性
    alpha=0.0,            # PML參數α
    eta=0.0,             # PML參數η
    pml_profile=mp.PMLProfile(1.0, 1.0)  # PML分佈
)]
```

## 完美導體邊界

### 完美電導體

```python
# 完美電導體邊界（E = 0）
sim = mp.Simulation(...)
sim.set_boundary(mp.Low, mp.X, mp.Metallic)
sim.set_boundary(mp.High, mp.X, mp.Metallic)
```

### 完美磁導體

```python
# 完美磁導體邊界（H = 0）
sim = mp.Simulation(...)
sim.set_boundary(mp.Low, mp.X, mp.Magnetic)
sim.set_boundary(mp.High, mp.X, mp.Magnetic)
```

### 混合邊界條件

```python
# x方向PML，y方向金屬
pml_layers = [mp.PML(thickness=1.0, direction=mp.X)]

sim = mp.Simulation(..., boundary_layers=pml_layers)
sim.set_boundary(mp.Low, mp.Y, mp.Metallic)
sim.set_boundary(mp.High, mp.Y, mp.Metallic)
```

## 週期邊界

### 布拉格週期邊界

```python
# 布拉格週期邊界
k_point = mp.Vector3(0.1, 0, 0)  # 布拉格向量

sim = mp.Simulation(..., k_point=k_point)
```

### 週期邊界自動重複

```python
# 啟用週期性（默認）
sim = mp.Simulation(..., k_point=k_point, ensure_periodicity=True)

# 禁用週期性
sim = mp.Simulation(..., k_point=k_point, ensure_periodicity=False)
```

### 二維k_point

```python
# 2D仿真中的非零k_z
k_point = mp.Vector3(0, 0, 0.1)

# 使用2D單元格
sim = mp.Simulation(..., k_point=k_point, kz_2d="complex")

# 使用3D單元格
sim = mp.Simulation(..., k_point=k_point, kz_2d="3d")

# 使用實/虛單元格
sim = mp.Simulation(..., k_point=k_point, kz_2d="real/imag")
```

## 吸收邊界

### 基本吸收器

```python
# 吸收邊界（替代PML）
absorber_layers = [mp.Absorber(thickness=1.0)]

sim = mp.Simulation(..., boundary_layers=absorber_layers)
```

### 吸收器參數

```python
# 完整吸收器參數
absorber_layers = [mp.Absorber(
    thickness=1.0,      # 吸收器厚度
    sigma=1.0,           # 電導率
    direction=mp.X,     # 方向
    side=mp.High         # 側邊
)]
```

## 邊界條件選擇

### 吸收邊界

**使用場景：**
1. 開導仿真
2. 散射問題
3. 開域邊界
4. 時域仿真

**優點：**
- 高效吸收
- 最小反射

**缺點：**
- 增加計算量
- 與散材料不穩定

### 完美導體邊界

**使用場景：**
1. 腔體腔體
2. 週期結構
3. 諯射器
4. 理想化邊界

**優點：**
- 不增加計算量
- 完美反射

**缺點：**
- 會產生反射
- 不適用於散射問題

### 週期邊界

**使用場景：**
1. 光子晶體
2. 週期波導
3. 布拉格結構
4. 布散態分析

**優點：**
- 模擬無限結構
- 適用於帶散計算

**缺點：**
- 只能用於週期結構
- 可能產生帶疊模式

### 吸收邊界

**使用場景：**
1. 散射材料仿真
2. 激散材料不穩定
3. PML與材料衝突

**優點：**
- 比PML更穩定
- 簡用於各種材料

**缺點：**
- 吸收效率較低
- 需要調整參數

## PML最佳實踐

### PML厚度

```python
# 基本規則：2-3個波長
wavelength = 1.0 / frequency  # 波長
pml_thickness = 2.5 * wavelength

pml_layers = [mp.PML(pml_thickness)]
```

### PML與幾何

```python
# 確保PML覆蓋幾何
cell = mp.Vector3(16, 8, 0)
pml_thickness = 1.0
pml_layers = [mp.PML(pml_thickness)]

# 幾何在PML內部
geometry = [mp.Block(size=mp.Vector3(14, 6, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=12))]
```

### PML與頻帶材料

```python
# 頻帶材料與PML重疊可能不穩定
# 解決方法：使用吸收器

# 使用吸收器代替PML
absorber_layers = [mp.Absorber(thickness=1.0, sigma=1.0)]
```

### PML性能

```python
# 使用PML分佈優化性能
pml_profile = mp.PMLProfile(1.0, 1.0)  # (α, η)
pml_layers = [mp.PML(thickness=1.0, pml_profile=pml_profile)]
```

## 邊界與對稱性

### 對稱邊界

```python
# Y方向對稱結構
symmetries = [mp.Mirror(mp.Y)]

# PML邊界（自動對稱）
pml_layers = [mp.PML(1.0)]

sim = mp.Simulation(..., symmetries=symmetries, boundary_layers=pml_layers)
```

### 對稱PML

```python
# 對稱PML可能導致問題
# 解決方法：在非對稱方向使用PML

# 只在x方向使用PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.X)]
```

## 邊界與源

### 源與邊界距離

```python
# 源應該遠離邊界至少1個單位距
cell = mp.Vector3(16, 8, 0)
pml_layers = [mp.PML(1.0)]

# 源在PML內部（正確）
sources = [mp.Source(..., center=mp.Vector3(-7, 0))]

# 源太靠近邊界（錯誤）
sources = [mp.Source(..., center=mp.Vector3(-7.5, 0))]
```

### 源與週期邊界

```python
# 週期邊界
k_point = mp.Vector3(0.1, 0, 0)

# 源位置（考慮週期性）
sources = [mp.Source(..., center=mp.Vector3(0, 0, 0))]
```

## 邊界與監控器

### 監控器位置

```python
# 監控器應該在PML外部
cell = mp.Vector3(16, 8, 0)
pml_layers = [mp.PML(1.0)]

# 監控器在PML外部（正確）
refl_fr = mp.FluxRegion(center=mp.Vector3(-5, 0), size=mp.Vector3(0, 2))

# 監控器在PML內部（錯誤）
refl_fr = mp.FluxRegion(center=mp.Vector3(-7.5, 0), size=mp.Vector3(0, 2))
```

### 監控器尺寸

```python
# 監控器應該覆蓋整個模式
waveguide_width = 1.0

# 監控器尺寸應該大於波導寬度
flux_size = 2.0 * waveguide_width
refl_fr = mp.FluxRegion(center=mp.Vector3(-5, 0), size=mp.Vector3(0, flux_size))
```

## 邊界故障排除

### PML反射

**原因：**
1. PML厚度不足
2. 解析度太低
3. 頻帶材料不穩定

**解決方法：**
1. 增加PML厚度（2-3個波長）
2. 增加解析度
3. 使用吸收器代替PML

### 邊界反射

**原因：**
1. 源太靠近邊界
2. 監控器在邊界上
3. 邊界條件不正確

**解決方法：**
1. 將源和監控器遠離邊界
2. 檢查邊界條件設置
3. 增加PML厚度

### 週期邊界問題

**原因：**
1. 結構不週期
2. k_point不正確
3. 對稱性不一致

**解決方法：**
1. 檢查結構週期性
2. 確認k_point正確
3. 檢查對稱性設置

### 仿真不穩定

**原因：**
1. PML與頻帶材料衝突
2. Courant因子太大
3. 邊界條件不正確

**解決方法：**
1. 使用吸收器代替PML
2. 減小Courant因子
3. 檢查邊界條件

## 高級邊界條件

### PML對稱性

```python
# R對稱性（減少反射）
pml_layers = [mp.PML(thickness=1.0, R_asymptote=True)]

# K對稱性（減少反射）
pml_layers = [mp.PML(thickness=1.0, K_asymptote=True)]
```

### 複雜邊界條件

```python
# 混合邊界條件
boundary_layers = [
    mp.PML(thickness=1.0, direction=mp.X),
    mp.Absorber(thickness=1.0, direction=mp.Y),
    mp.PML(thickness=1.0, direction=mp.Z)
]
```

### 動態邊界條件

```python
# 根據仿真類型設置邊界條件
sim = mp.Simulation(...)

# 動態設置邊界條件
sim.set_boundary(mp.Low, mp.X, mp.Metallic)
sim.set_boundary(mp.High, mp.X, mp.Metallic)
```

## 邊界與計算域

### 計算域大小

```python
# PML在計算域內部
cell = mp.Vector3(16, 8, 0)  # 計算域大小
pml_layers = [mp.PML(1.0)]  # PML在計算域內部

# 實際仿真域大小 = cell + 2*pml_thickness
```

### 監控器位置

```python
# 監控器應該在計算域內部
cell = mp.Vector3(16, 8, 0)
pml_layers = [mp.PML(1.0)]

# 監控器在計算域內部（正確）
refl_fr = mp.FluxRegion(center=mp.Vector3(-5, 0), size=mp.Vector3(0, 2))

# 監控器在PML內部（仍然有效）
refl_fr = mp.FluxRegion(center=mp.Vector3(-7.5, 0), size=mp.Vector3(0, 2))
```

## 邊界與維度

### 邊界條件與維度

```python
# PML增加計算量
pml_layers = [mp.PML(1.0)]  # 增加約10-20%計算量

# 金屬邊界不增加計算量
sim.set_boundary(mp.Low, mp.X, mp.Metallic)  # 不增加計算量
```

### 優化邊界條件

```python
# 只在需要的方向使用PML
pml_layers = [mp.PML(thickness=1.0, direction=mp.X)]

# 其他方向使用金屬邊界
sim = mp.Simulation(..., boundary_layers=pml_layers)
sim.set_boundary(mp.Low, mp.Y, mp.Metallic)
sim.set_boundary(mp.High, mp.Y, mp.Metallic)
```

## 邊界與精度

### PML與精度

```python
# PML厚度影響吸收精度
# 基本規則：2-3個波長
wavelength = 1.0 / frequency
pml_thickness = 2.5 * wavelength
```

### 解析度與PML

```python
# 解析度影響PML性能
resolution = 20  # 高解析度
pml_layers = [mp.PML(1.0)]  # PML在高解析度下工作更好
```

### 邊界與收斂

```python
# 收斂要求
# - PML厚度：2-3個波長
# - 源距離：>1個單位距
# - 監控器距離：>1個單位距
# - 解析度：10-20像素/波長
```