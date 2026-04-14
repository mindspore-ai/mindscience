# Materials in Meep

Meep物質系統描述了Maxwell方程中的相對介電常數ε和磁導率μ的確定。

## 基本材料定義

### 簡帶材料

```python
mp.Medium(epsilon=12)  # 相對介電常數
```

### 頻帶和磁導率

```python
mp.Medium(epsilon=3.4, mu=1.0)  # ε和μ
```

### 頻帶材料屬

後面的對象優先：

```python
geometry = [mp.Block(material=mp.Medium(epsilon=12), ...),
            mp.Block(material=mp.Medium(epsilon=4), ...)]  # 第二個覆蓋第一個
```

## 頴帶材料

### 電導率

```python
# 電導材料（損耗）
mp.Medium(epsilon=3.4, D_conductivity=0.5)

# 磁導材料（增益）
mp.Medium(epsilon=3.4, D_conductivity=-0.1)
```

注意：Meep中的電導率定義與教科書略有不同：
- Meep使用D = σ/ε₀
- 從SI單位轉換：D_Meep = (2π * unit_distance / c) * σ_SI / ε₀

### 頴帶複數ε

對窄帶寬計算，使用電導率在特定頻率模擬複數ε：

```python
# 在頻率f = 0.42處實現ε = 3.4 + 0.101j
import math
sigma = 2 * math.pi * 0.42 * 0.101 / 3.4
material = mp.Medium(epsilon=3.4, D_conductivity=sigma)
```

## 頴帶材料

### 洛倫斯坦極化模型

```python
susceptibility = mp.LorentzianSusceptibility(frequency=0.3,
                                            gamma=0.1,
                                            sigma=0.5)
material = mp.Medium(epsilon=3.4, susc=susceptibility)
```

參數說明：
- `frequency`: 共振頻率ω₀
- `gamma`: 阻尼係數γ
- `sigma`: 極強度σ

對應的頻率依賴：
ε(ω) = ε∞ + σ²/[(ω₀² - ω²) + iγω]

### 德魮模型（金屬）

```python
susceptibility = mp.DrudeSusceptibility(gamma=0.1, sigma=0.5)
material = mp.Medium(epsilon=1.0, susc=susceptibility)
```

對應的頻率依賴：
ε(ω) = ε∞ - σ²/[ω(ω + iγ)]

### 多重共振

```python
susceptibilities = [
    mp.LorentzianSusceptibility(frequency=0.3, gamma=0.1, sigma=0.5),
    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.2, sigma=0.3)
]
material = mp.Medium(epsilon=3.4, susc=susceptibilities)
```

### Sellmeier方程轉換

對純實相對介電常數（無損耗）的Sellmeier方程：
n²(λ) = 1 + Σ[Bᵢλ²/(λ² - Cᵢ)]

轉換為洛倫茲模型：
- ω₀² = 2πc/Cᵢ
- σ² = Bᵢω₀²

## 非線性材料

### 克爾效應（χ³）

```python
# 克爾非線性
material = mp.Medium(epsilon=2.0, chi3=0.1)
```

克爾效應：ε = ε₀ + χ₃|E|²

### 鮡克爾斯效應（χ²）

```python
# 對角對角線性
material = mp.Medium(epsilon=2.0, chi2_diag=[0.1, 0.1, 0.1])
```

## 磁帶材料

### 各向異性材料

```python
# 各向異性張量
epsilon_tensor = [[12, 0, 0],
                   [0, 12, 0],
                   [0, 0, 12]]
material = mp.Medium(epsilon=epsilon_tensor)
```

### 磁帶張量

```python
# 磁帶ε張量
susceptibility = mp.LorentzianSusceptibility(
    frequency=0.3,
    gamma=0.1,
    sigma=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
)
```

## 磁帶材料

### 克爾效應

```python
# 磁帶克爾非線性
material = mp.Medium(epsilon=2.0, mu=1.0, chi3=0.1)
```

### 磁帶洛倫茲共振

```python
# 磁帶洛倫茲極化
susceptibility = mp.LorentzianSusceptibility(frequency=0.3, gamma=0.1, sigma=0.5)
material = mp.Medium(epsilon=2.0, mu=1.0, susc=susceptibility)
```

## 飽帶材料

### 實帶材料

Meep支持飽和材料，用於模擬激光等系統：

```python
# 兩能級系統
levels = [
    mp.MultilevelAtom(level=0, pumping_rate=0.1),
    mp.MultilevelAtom(level=1, decay_rate=0.01)
]

# 遷帶遷移
transition = mp.Transition(from_level=0, to_level=1,
                           frequency=0.3,
                           gamma=0.001,
                           sigma=0.5)

# 飽帶材料
material = mp.Medium(
    epsilon=2.0,
    multilevel_atoms=levels,
    transitions=[transition],
    total_population_density=1e20
)
```

## 磁帶材料

### 旋電性洛倫茲模型

```python
# 旋電性洛倫茲極化
susceptibility = mp.GyrotropicLorentzianSusceptibility(
    frequency=0.3,
    gamma=0.1,
    sigma=0.5,
    bias=mp.Vector3(0, 0, 0.1)  # 旋轉軸
)
material = mp.Medium(epsilon=2.0, susc=susceptibility)
```

### 旋磁飽和偶極子模型

```python
# 旋磁飽和偶極子（Landau-Lifshitz-Gilbert）
susceptibility = mp.GyrotropicSaturatedSusceptibility(
    frequency=0.3,
    gamma=0.1,
    sigma=0.5,
    bias=mp.Vector3(0, 0, 0.1)  # 磁化方向
)
material = mp.Medium(epsilon=2.0, mu=1.0, susc=susceptibility)
```

## 材料庫

Meep提供預定義的材料庫，包含常見光學材料的複數折射率數擬合。

### 使用材料庫

```python
from meep.materials import Si, Au, Al, SiO2, GaAs

# 使用預定義材料
geometry = [mp.Block(material=Si, ...)]
```

### 可用材料

**半導體：**
- Si（晶體矽）
- aSi（非晶矽）
- GaAs（砷化鎵）
- AlAs（砷化�）
- InP（磷化銦）
- Ge（鍺）

**電介質：**
- SiO2（二氧化矽）
- Al2O3（氧化�）
- TiO2（二氧化�）
- HfO2（氧化鉿）

**金屬：**
- Au（金）
- Ag（銀）
- Al（�）
- Cu（銅）

**玻璃：**
- BK7（硼矽酸鹽玻璃）
- FusedSilica（熔融石英）

### 檢查材料屬性

```python
from meep.materials import SiO2
import numpy as np

# 檢查有效頻率範圍
print(SiO2.valid_freq_range)
# FreqRange(min=0.0, max=10.0)

# 計查特定頻率的介電常數
wavelength = 1.55  # μm
epsilon = SiO2.epsilon(1/wavelength)
print(epsilon)
# [[3.479+0j, 0+0j, 0+0j],
#  [0+0j, 3.479+0j, 0+0j],
#  [0+0j, 0+0j, 3.479+0j]]
```

### 單位縮放

材料庫的擬合參數基於1 μm單位距。對於其他單位距：

```python
from meep.materials import um_scale

# 如果單位距是100 nm
um_scale = 0.1 * um_scale
```

## 數值穩定性

### 洛倫斯坦極化

高頻洛倫茲共振可能導致數值不穩定：
- �件件：增加解析度
- �件件：減小Courant因子
- �件件：使用不同的模型函數

穩定條件：ω₀ < π/(Courant * Δt)

### PML與頻帶材料

頻帶材料與PML重疊可能導致不穩定：
- �件件：用吸收器代替PML
- �件件：減小PML厚度

### 飽帶材料穩定性

飽和材料需要：
- 確保操作在穩定態
- 使用窄帶源
- 等待瞬態衰減

## 材料參數

### 檢取介電常數張量

```python
# 在特定頻率和位置獲取介電常數張量
freq = 0.15
pt = mp.Vector3(0, 0, 0)
epsilon_tensor = sim.get_epsilon_point(pt, freq)
```

### 檇取磁導率張量

```python
# 在特定頻率和位置獲取磁導率張量
freq = 0.15
pt = mp.Vector3(0, 0, 0)
mu_tensor = sim.get_mu_point(pt, freq)
```

## 材料函數

### 位置依賴材料

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
    if p.x < 0:
        return 12.0
    else:
        return 4.0

sim = mp.Simulation(..., epsilon_func=epsilon_func)
```

### 從HDF5導入

```python
# 從HDF5文件導入介電常數分佈
sim = mp.Simulation(..., epsilon_input_file="epsilon.h5")
```

## 最佳實踐

### 材料選擇

1. **簡單問題**：使用頻帶材料
2. **窄帶問題**：使用電導率
3. **寬帶問題**：使用洛倫茲模型
4. **金屬**：使用德魯模型
5. **激光**：使用飽和材料

### 解析度要求

- 頴帶材料：8-10像素/波長
- 頴帶材料：10-15像素/波長
- 高精度：20+像素/波長

### Courant因子調整

頻帶材料可能需要減小Courant因子：
```python
sim = mp.Simulation(..., Courant=0.4)  # 頹帶從0.5減到0.4
```

### 材料庫使用

- 檢查有效頻率範圍
- 確認單位距縮放
- 必要時移除不必要的洛倫茲項