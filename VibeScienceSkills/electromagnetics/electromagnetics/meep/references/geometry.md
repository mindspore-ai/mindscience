# Geometric Objects in Meep

幾何對象用於定義仿真中的幾何結構。

## 基本幾何對象

### Block（塊）

長方體（平行六面體）：

```python
# 基本塊
mp.Block(size=mp.Vector3(10, 5, mp.inf),
          center=mp.Vector3(0, 0, 0),
          material=mp.Medium(epsilon=12))

# 無旋轉的塊
rotated_block = mp.Rotate2(mp.Vector3(0, 0, 0),  # 旋轉中心
                         math.pi/4,  # 旋轉角度
                         mp.Vector3(0, 0, 1))  # 旋轉軸
rotated_block = rotated_block + mp.Block(size=mp.Vector3(10, 5, mp.inf),
                                       material=mp.Medium(epsilon=12))
```

參數：
- `size`: 尺寸（Vector3）
- `center`: 中心位置（Vector3）
- `material`: 材料（Medium）
- `e1`, `e2`, `e3`: 坐標向量（默認為坐標軸）

### Cylinder（圓柱）

```python
# 基本圓柱
mp.Cylinder(radius=2, height=mp.inf,
            center=mp.Vector3(0, 0, 0),
            axis=mp.Z,
            material=mp.Medium(epsilon=12))

# 有限高度圓柱
mp.Cylinder(radius=2, height=5,
            center=mp.Vector3(0, 0, 0),
            axis=mp.Z,
            material=mp.Medium(epsilon=12))

# 沿軸圓柱（y方向）
mp.Cylinder(radius=2, height=mp.inf,
            center=mp.Vector3(0, 0, 0),
            axis=mp.Y,
            material=mp.Medium(epsilon=12))
```

參數：
- `radius`: 半徑
- `height`: 高度（mp.inf表示無限）
- `center`: 中心位置
- `axis`: 軸線方向（mp.X, mp.Y, mp.Z）
- `material`: 材料

### Sphere（球體）

```python
# 基本球體
mp.Sphere(radius=1.5,
          center=mp.Vector3(0, 0, 0),
          material=mp.Medium(epsilon=12))
```

參數：
- `radius`: 半徑
- `center`: 中心位置
- `material`: 材料

### Ellipsoid（橢球體）

```python
# 基本橢球體
mp.Ellipsoid(size=mp.Vector3(3, 2, 1),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=12))
```

參數：
- `size`: 橫寸（Vector3）
- `center`: 中心位置
- `material`: 材料

## 高級幾何對象

### Wedge（楔形）

```python
# 楔形
mp.Wedge(vertex1=mp.Vector3(0, 0, 0),
          vertex2=mp.Vector3(1, 0, 0),
          vertex3=mp.Vector3(0, 1, 0),
          height=mp.inf,
          material=mp.Medium(epsilon=12))
```

參數：
- `vertex1`, `vertex2`, `vertex3`: 三角形頂點
- `height`: 高度（垂直於三角形平面）
- `material`: 材料

### Cone（圓錐）

```python
# 圓錐
mp.Cone(radius=2, height=5,
          center=mp.Vector3(0, 0, 0),
          axis=mp.Z,
          material=mp.Medium(epsilon=12))
```

參數：
- `radius`: 底面半徑
- `height`: 高度
- `center`: 中心位置
- `axis`: 軸線方向
- `material`: 材料

### Prism（稜柱）

```python
# 稜柱（多邊形柱）
vertices = [mp.Vector3(0, 0, 0),
             mp.Vector3(1, 0, 0),
             mp.Vector3(0.5, 0.866, 0)]

mp.Prism(vertices=vertices,
          height=mp.inf,
          center=mp.Vector3(0, 0, 0),
          axis=mp.Z.
          material=mp.Medium(epsilon=12))
```

參數：
- `vertices`: 底面頂點列表
- `height`: 高度
- `center`: 中心位置
- `axis`: 軸線方向
- `material`: 材料

## 幾何變換

### 旋轉

**2D旋轉：**

```python
# 2D旋轉（繞z軸旋轉）
rotated_obj = mp.Rotate2(mp.Vector3(0, 0, 0),  # 旋轉中心
                          math.pi/4,  # 旋轉角度（弧度）
                          mp.Vector3(0, 0, 1))  # 旋轉軸

# 應用旋轉
geometry = [rotated_obj + mp.Block(size=mp.Vector3(10, 5, mp.inf),
                                     material=mp.Medium(epsilon=12))]
```

**4D旋轉：**

```python
# 4D旋轉（3D旋轉）
rotated_obj = mp.Rotate4(mp.Vector3(0, 0, 0),  # 旋轉中心
                          math.pi/4,  # 旋轉角度（弧度）
                          mp.Vector3(0, 0, 1))  # 旋轉軸
```

### 平移

```python
# 平移
translated_obj = mp.Translate(mp.Vector3(5, 0, 0))

# 應用平移
geometry = [translated_obj + mp.Block(size=mp.Vector3(10, 5, mp.inf),
                                        material=mp.Medium(epsilon=12))]
```

### 縮換

```python
# 縮換矩陣
matrix = mp.Matrix(mp.Vector3(1, 0, 0),
                  mp.Vector3(0, 1, 0),
                  mp.Vector3(0, 0, 1))

scaled_obj = mp.Scale(matrix)

# 懲用縮放
geometry = [scaled_obj + mp.Block(size=mp.Vector3(10, 5, mp.inf),
                                  material=mp.Medium(epsilon=12))]
```

### 組合變換

```python
# 組合變換
transform = mp.Translate(mp.Vector3(5, 0, 0)) + \
              mp.Rotate2(mp.Vector3(0, 0, 0), math.pi/4, mp.Vector3(0, 0, 1))

geometry = [transform + mp.Block(size=mp.Vector3(10, 5, mp.inf),
                                 material=mp.Medium(epsilon=12))]
```

## 複雜幾何

### 多個對象

```python
geometry = [
    mp.Block(size=mp.Vector3(10, 5, mp.inf),
             center=mp.Vector3(-2.5, 0, 0),
             material=mp.Medium(epsilon=12)),
    mp.Block(size=mp.Vector3(1, 10, mp.inf),
             center=mp.Vector3(3.5, 0, 0),
             material=mp.Medium(epsilon=12))
]
```

### 對疊規則

後面的對象優先：

```python
geometry = [
    mp.Block(size=mp.Vector3(10, 10, mp.inf),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=12)),  # 背景
    mp.Block(size=mp.Vector3(5, 5, mp.inf),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=4))  # 覆蓋前景
]
```

## 複雜結構

### 波導

```python
# 直波導
geometry = [mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=12))]

# 彎波導
geometry = [
    mp.Block(size=mp.Vector3(12, 1, mp.inf),
             center=mp.Vector3(-2.5, -3.5, 0),
             material=mp.Medium(epsilon=12)),
    mp.Block(size=mp.Vector3(1, 12, mp.inf),
             center=mp.Vector3(3.5, 2, 0),
             material=mp.Medium(epsilon=12))
]
```

### 光子晶體

```python
# 一維光子晶體（布拉格反射鏡）
a = 1.0  # 晶格常數
radius = 0.2
geometry = [mp.Cylinder(radius=radius, height=mp.inf,
                     center=mp.Vector3(0, 0, 0),
                     axis=mp.Z,
                     material=mp.Medium(epsilon=12))]

# 多個圓柱形成二維晶格
for i in range(-5, 6):
    for j in range(-5, 6):
        geometry.append(mp.Cylinder(radius=radius, height=mp.inf,
                               center=mp.Vector3(i*a, j*a, 0),
                               axis=mp.Z,
                               material=mp.Medium(epsilon=12)))
```

### 諌射體

```python
# 褓射體
geometry = [mp.Sphere(radius=1.0,
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=2.0))]
```

### 腔體

```python
# 環形腔體
geometry = [
    mp.Block(size=mp.Vector3(10, 10, mp.inf),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=12)),
    mp.Block(size=mp.Vector3(8, 8, mp.inf),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=1))  # 空體空腔
]
```

## 複雜材料

### 不同材料

```python
geometry = [
    mp.Block(size=mp.Vector3(5, 10, mp.inf),
             center=mp.Vector3(-2.5, 0, 0),
             material=mp.Medium(epsilon=12)),  # 矽材料1
    mp.Block(size=mp.Vector3(5, 10, mp.inf),
             center=mp.Vector3(2.5, 0, 0),
             material=mp.Medium(epsilon=4))  # 矽材料2
]
```

### 材料庫材料

```python
from meep.materials import Si, Au, SiO2

geometry = [
       mp.Block(size=mp.Vector3(5, 10, mp.inf),
             center=mp.Vector3(-2.5, 0, 0),
             material=Si),  # 矽
    mp.Block(size=mp.Vector3(5, 10, mp.inf),
             center=mp.Vector3(2.5, 0, 0),
             material=Au)  # 金
]
```

### 複雜材料

```python
# 複雜材料（洛倫茲）
susceptibility = mp.LorentzianSusceptibility(frequency=0.3, gamma=0.1, sigma=0.5)
material = mp.Medium(epsilon=3.4, susc=susceptibility)

geometry = [mp.Block(size=mp.Vector3(10, 5, mp.inf),
                     material=material)]
```

## 幾何與網格

### 網像素平滑

Meep使用亞像素平滑提高精度：

```python
# 啟用亞像素平滑（默認）
sim = mp.Simulation(..., eps_averaging=True)

# 禁用亞像素平滑
sim = mp.Simulation(..., eps_averaging=False)
```

亞像素平滑參數：
- `subpixel_tol`: 積分容差（默認1e-4）
- `subpixel_maxeval`: 最大函數評估次數（默認1e5）

```python
sim = mp.Simulation(..., 
                    eps_averaging=True,
                    subpixel_tol=1e-4,
                    subpixel_maxeval=1e5)
```

### 解析度考慮

幾何特徵的解析度要求：
- **平滑邊界**：10-15像素/特徵尺寸
- **細節特徵**：20+像素/特徵尺寸
- **曲邊界**：15-20像素/曲率半徑

## 幾何與對稱性

### 對稱幾何

利用對稱性減少計算量：

```python
# Y方向對稱結構
symmetries = [mp.Mirror(mp.Y)]

geometry = [
    mp.Block(size=mp.Vector3(10, 5, mp.inf),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=12))
]
```

### 對稱旋轉

```python
# 90度旋轉對稱
symmetries = [mp.Rotate4(mp.Vector3(0, 0, 0), math.pi/2, mp.Vector3(0, 0, 1))]

geometry = [mp.Cylinder(radius=1, height=mp.inf,
                     center=mp.Vector3(2, 0, 0),
                     axis=mp.Z,
                     material=mp.Medium(epsilon=12))]
```

### 對稱相移

```python
# Y方向對稱
symmetries = [mp.Mirror(mp.Y)]

# 創用對稱
sim = mp.Simulation(..., symmetries=symmetries)
```

## 幾何與邊界

### 幾何與PML

確保PML覆蓋幾何：

```python
# PML厚度
dpml = 1.0
pml_layers = [mp.PML(dpml)]

# 幾何在PML內部
cell = mp.Vector3(16, 8, 0)
geometry = [mp.Block(size=mp.Vector3(14, 4, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=12))]
```

### 幾何與週期邊界

週期邊界自動重複幾何：

```python
# 週期邊界
k_point = mp.Vector3(0.1, 0, 0)  # 布拉格向量

# 啟用週期性（默認）
sim = mp.Simulation(..., k_point=k_point, ensure_periodicity=True)

# 禁用週期性
sim = mp.Simulation(..., k_point=k_point, ensure_periodicity=False)
```

## 幾何函數

### 材料函數

```python
def material_func(p):
    # p是Vector3位置
    if p.x < 0:
        return mp.Medium(epsilon=12)
    else:
        return mp.Medium(epsilon=4)

sim = mp.Simulation(..., material_function=material_func)
```

### 介電常數函數

```python
def epsilon_func(p):
    # 返回標量介電常數
    r = math.sqrt(p.x**2 + p.y**2)
    if r < 5:
        return 12.0
    else:
        return 1.0

sim = mp.Simulation(..., epsilon_func=epsilon_func)
```

### 從HDF5導入

```python
# 從HDF5文件導入介電常數分佈
sim = mp.Simulation(..., epsilon_input_file="epsilon.h5")
```

## 最佳實踐

### 幾何設計

1. **簡單結構**：使用基本幾何對象
2. **複雜結構**：組合多個對象
3. **週期結構**：使用週期邊界
4. **參數化結構**：使用幾何函數

### 對疊規則

1. **背景優先**：先定義背景材料
2. **特徵優先**：後定義特徵材料
3. **測試覆蓋**：使用簡單幾何測試

### 變算優化

1. **利用對稱性**：減少網格大小
2. **亞像素平滑**：提高精度
3. **合理解析度**：平衡精度和速度

### 幾何驗證

1. **檢查尺寸**：確保幾何在網格內
2. **檢查材料**：驗證材料屬性
3. **檢查對稱性**：確保對稱性一致

## 故障排除

### 幾何不顯示

**原因：**
1. 幾何在網格外
2. 幾何尺寸為0
3. 材料不正確

**解決方法：**
1. 檢查幾何位置和尺寸
2. 確認幾何在網格內
3. 驗證材料定義

### 幾何邊界不正確

**原因：**
1. 幾何與邊界重疊
2. PML厚度不足
3. 對稱性不一致

**解決方法：**
1. 調查幾何和邊界位置
2. 增加PML厚度
3. 確認對稱性一致

### 材料不應用

**原因：**
1. 材料屬疊順序錯誤
2. 幾何尺寸為0
3. 材料函數錯誤

**解決方法：**
1. 檢查幾何列表順序
2. 確認幾何尺寸非零
3. 驗證材料函數

### 解析度不足

**原因：**
1. 解析度太低
2. 幾何特徵太小

**解決方法：**
1. 增加解析度
2. 檢查特徵尺寸
3. 使用亞像素平滑