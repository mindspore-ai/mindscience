# Sources in Meep

源用於在FDTD網格中激勵電磁場。

## 源型

### 連續波源

```python
# 基本連續波源
mp.ContinuousSource(frequency=0.15)

# 帶漸升時間
mp.ContinuousSource(frequency=0.15, width=20)

# 帶相位
mp.ContinuousSource(frequency=0.15, phase=math.pi/2)
```

參數：
- `frequency`: 源頻率（單位：2πc）
- `width`: 升時間（減少高頻內容）
- `phase`: 相位偏移

### 高斯脈衝

```python
# 基本高斯脈衝
mp.GaussianSource(fcen=0.15, fwidth=0.1)

# 帶中心頻率和寬度
mp.GaussianSource(fcen=0.15, fwidth=0.1, start_time=0, end_time=100)
```

參數：
- `fcen`: 中心頻率
- `fwidth`: 頻率寬度（標準差）
- `start_time`: 開始時間
- `end_time`: 結束時間

頻譜範圍：[fcen - fwidth, fcen + fwidth]

### 自定義源

```python
def my_source_func(t):
    # t是時間
    return math.sin(2 * math.pi * 0.15 * t)

mp.CustomSource(src_func=my_source_func)
```

## 源配置

### 基定義

```python
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0),
                     size=mp.Vector3(0, 1))]
```

參數：
- `src_func`: 源時間函數
- `component`: 場場分量
- `center`: 源位置
- `size`: 源尺寸（點源時為0）

### 場場分量

**電場分量：**
- `mp.Ex`: x方向電場
- `mp.Ey`: y方向電場
- `mp.Ez`: z方向電場

**磁場分量：**
- `mp.Hx`: x方向磁場
- `mp.Hy`: y方向磁場
- `mp.Hz`: z方向磁場

**導出場分量：**
- `mp.Dx`, `mp.Dy`, `mp.Dz`: 電位移場
- `mp.Bx`, `mp.By`, `mp.Bz`: 磁感應場

### 點源

```python
# 點源
sources = [mp.Source(..., center=mp.Vector.x(-7, 0, 0))]
```

### 線源

```python
# 線源（填充整個平面）
sources = [mp.Source(..., center=mp.Vector3(0, 0, 0),
                     size=mp.Vector3(10, 10, 0))]
```

### 線源

```python
# 線源（填充整個體積）
sources = [mp.Source(..., center=mp.Vector3(0, 0, 0),
                     size=mp.Vector3(10, 10, 10))]
```

## 特殊源

### 本模態源

用於激勵波導模態：

```python
# 獲取波導模態
k = mp.Vector3(0.15, 0, 0)  # 傳播常數
mode = sim.get_eigenmode(k, mp.Ez, mp.Vector3(0, 0), mp.Vector3(0, 1))

# 使用本模態源
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0),
                     eigenmode=mode)]
```

### 高斯光束源

**3D高斯光束：**

```python
# 3D高斯光束
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0, 0),
                     gaussian_beam=mp.GaussianBeam3DSource(
                         w0=1.0,  # 束腰半徑
                         k0=mp.Vector3(0, 0, 0.15),  # 傳播向量
                         z0=0  # 束腰位置
                     ))]
```

**2D高斯光束：**

```python
# 2D高斯光束
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0, 0),
                     gaussian_beam=mp.GaussianBeam2DSource(
                         w0=1.0,  # 束腰半徑
                         k0=mp.Vector3(0, 0.15, 0),  # 傳播向量
                         z0=0  # 束腰位置
                     ))]
```

## 源時序函數

### 基本時序

Meep使用雙曲正切函數進行平滑升啟：

```python
# 默升啟（tanh函數）
mp.ContinuousSource(frequency=0.15, width=20)
```

升啟函數：f(t) = 0.5 * (1 + tanh(2π * (t - t₀) / width))

### 自定義時序

```python
def my_source_func(t):
    # 自定義時序函數
    if t < 10:
        return math.sin(2 * math.pi * 0.15 * t) * (t / 10)
    else:
        return math.sin(2 * math.pi * 0.15 * t)

sources = [mp.Source(mp.CustomSource(src_func=my_source_func),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]
```

## 多源

### 多個源

```python
sources = [
    mp.Source(mp.ContinuousSource(frequency=0.15),
              component=mp.Ez,
              center=mp.Vector3(-7, 0)),
    mp.Source(mp.ContinuousSource(frequency=0.15, phase=math.pi/2),
              component=mp.Ez,
              center=mp.Vector3(7, 0))
]
```

### 時序多源

```python
# 第一個源在t=100時關閉
source1 = mp.Source(mp.ContinuousSource(frequency=0.15, end_time=100),
                     component=mp.Ez,
                     center=mp. Vector3(-7, 0))

# 第二個源在t=100時開啟
source2 = mp.Source(mp.ContinuousSource(frequency=0.15, start_time=100),
                     component=mp.Ez,
                     center=mp.Vector3(7, 0))

sources = [source1, source2]
```

## 源與邊界條件

### 源位置

**重要：** 源應該遠離邊界至少1個單位距：

```python
# 好的佈置
cell = mp.Vector3(16, 8, 0)
pml_layers = [mp.PML(1.0)]

# 源在PML內部
sources = [mp.Source(..., center=mp.Vector3(-7, 0))]  # 正確

# 源太靠近邊界
sources = [mp.Source(..., center=mp.Vector3(-7.5, 0))]  # 錀誤！
```

### 平面波源與PML

平面波源擴展到PML時，需要設置`is_integrated=True`：

```python
# 平面波源（填充整個網格）
sources = [mp.Source(mp.GaussianSource(fcen=0.15, fwidth=0.1, is_integrated=True),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0, 0),
                     size=mp.Vector3(16, 8, 0))]
```

## 源與對稱性

### 對稱源

源應該滿足對稱性條件：

```python
# Y方向對稱結構
symmetries = [mp.Mirror(mp.Y)]

# 源必須在對稱面上
sources = [mp.Source(..., center=mp.Vector3(-7, 0, 0))]  # 正確

# 源不在對稱面上
sources = [mp.Source(..., center=mp.Vector3(-7, 1, 0))]  # 錛誤！
```

### 對稱源

對稱源可以激勵對稱模態：

```python
# 對稱源
symmetries = [mp.Mirror(mp.Y, phase=-1)]

# 源在對稱面上
sources = [mp.Source(..., center=mp.Vector3(-7, 0, 0))]
```

## 源頻率

### 頻率單位

Meep中頻率單位為2πc：
- 頻率f = 0.15對應真空波長λ = 1/f ≈ 6.67 μm
- 如果單位距為1 μm，則f = 0.15對應真空波長6.67 μm

### 波長轉頻率

```python
# 從波長計算頻率
wavelength = 1.55  # μm
frequency = 1.0 / wavelength  # Meep單位

# 從頻率計算波長
frequency = 0.15
wavelength = 1.0 / frequency  # Meep單位
```

### 材料中的頻率

頻率在材料中的意義：
- 洛倫茲共振：frequency是共振頻率ω₀
- 德魯模型：frequency是特徵頻率
- 電導率：在特定頻率f處，電導率σ對應損耗tan(δ) = σ * f / ε

## 源耦合

### 點源耦合效率

點源對波導模態的耦合效率較低：

```python
# 低效率點源
sources = [mp.Source(..., center=mp.Vector3(-7, 0), size=mp.Vector3(0, 0))]
```

### 緬源耦合效率

使用線源或本模態源提高耦合效率：

```python
# 緬源（高效率）
sources = [mp.Source(..., center=mp.Vector3(-7, 0), size=mp.Vector3(0, 1))]

# 本模態源（最高效率）
mode = sim.get_eigenmode(k, mp.Ez, mp.Vector3(0, 0), mp.Vector3(0, 1))
sources = [mp.Source(..., eigenmode=mode)]
```

## 源極化

### 線極化

```python
# x方向極化
sources = [mp.Source(..., component=mp.Ex)]

# y方向極化
sources = [mp.Source(..., component=mp.Ey)]

# z方向極化
sources = [mp.Source(..., component=mp.Ez)]
```

### 磁場源

```python
# 磁場源
sources = [mp.Source(..., component=mp.Hx)]
```

### 極擇

**TM模態（橫磁波）：**
- 2D：Ez源
- 3D：Hx, Hy源

**TE模態（橫電波）：**
- 2D：Hz源
- 3D：Ex, Ey源

## 源幅度

### 源幅度控制

```python
# 使用幅度函數
def amplitude_func(t):
    return 1.0  # 單位幅度

sources = [mp.Source(mp.ContinuousSource(frequency=0.15, amplitude_func=amplitude_func),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]
```

### 源功率

源功率與幅度的平方成正比：
- 功率P ∝ |E|²
- 增加幅度需要考慮單位和歸一性

## 最佳實踐

### 源選擇

1. **頻譜分析**：使用高斯脈衝
2. **穩定態**：使用連續波源
3. **波導耦合**：使用本模態源
4. **自由空間**：使用高斯光束源
5. **散射問題**：使用平面波源

### 源參數設置

1. **頻率**'根據問題特徵頻率設置
2. **升時間**'使用寬度=10-20減少高頻內容
3. **相位**'根據需要設置相位偏移
4. **位置**'保持至少1單位距離邊界

### 源尺寸

1. **點源**'size=Vector3(0, 0, 0)
2. **線源**'一個方向為0，其他方向非零
3. **面源**'兩個方向為0，一個方向非零
4. **體源**'所有方向非零

### 對稱性

1. **檢查對稱性**'確保源滿足結構對稱性
2. **使用對稱源**'提高耦合效率
3. **減少計算量**'利用對稱性減少網格大小

### 疇帶源

1. **頻譜範圍**'確保覆蓋感興趣的頻率範圍
2. **脈衝寬度**'根據解析度設置fwidth
3. **運行時間**'確保足夠長以捕獲完整響應

## 故障排除

### 源不激勵模態

**原因：**
1. 源頻率在帶隙之外
2. 源極化不正確
3. 源位置錯誤

**解決方法：**
1. 檴查波導帶隙頻率
2. 檢查源極化方向
3. 確認源在波導內

### 高頻內容

**原因：**
1. 升時間太短
2. 源開啟/關閉太快

**解決方法：**
1. 增加width參數
2. 使用平滑升啟函數

### 源反射

**原因：**
1. 源太靠近邊界
2. 邊界條件不正確

**解決方法：**
1. 將源遠離邊界
2. 檢查PML設置

### 頻譜不正確

**原因：**
1. 頻譜範圍不匹配
2. 源時序不正確
3. 運尼太大

**解決方法：**
1. 擴大頻譜範圍
2. 增加運行時間
3. 減小頻率寬度