### class Fusiform
曲线方程为：
$y=4p(x-x_0)^2+c+\epsilon sin(\omega x+\phi)$  

这是一条叠加了正弦波的**开口向上**的抛物线，将该曲线沿一条水平线$y=y_{sym}$对称得到另一半开口向下的抛物线  

两条抛物线的交点标记为$x_{start}$和$x_{end}$，画图可以只画这个区间  
center和ratio是caption的时候用

在rules.json里该类fusiform的示例
```
{
    "type": "fusiform_1",
    "focal_length": 0.32,  # p
    "x_offset": 0.5,  # x_0
    "y_offset": 0.36,  # c
    "y_symmetric_axis": 0.48,  # y_sym
    "sin_params": [
        0.079,  # epsilon
        9.424,  # omega
        0  # phi
    ],
    "x_start": 0.267,
    "x_end": 0.732,
    "center": [
        0.5,
        0.48
    ],
    "ratio": 1.162,
    "special_info": "volution 2. ",
    "precision": 0.01
}
```

### class Fusiform_2
曲线方程为：
$y=(\frac{x-x_0}{4p})^{\frac{1}{m}}+y_0+\epsilon sin(\omega x+\phi)$  

这是一条叠加了正弦波的**开口向右**的抛物线（的上半部分）

在rules.json里该类fusiform的示例
```
{
    "type": "fusiform_2",
    "focal_length": 134.30,  # p
    "x_offset": 0.31,  # x_0
    "y_offset": 0.50,  # y_0
    "power": 3.60,  # m
    "x_symmetric_axis": 0.49,
    "sin_params": [
        0.022,
        14.50,
        3.14
    ],
    "x_start": 0.31,
    "x_end": 0.67,
    "center": [
        0.49,
        0.50
    ],
    "ratio": 1.46,
    "special_info": "volution 2. ",
    "precision": 0.01
}
```
根据该方程只能画出上半部分，可以采取这样的方式得到整个纺锤形
```Python
x = np.linspace(0, 1, 100)
x_left = x[: 50]
epsilon, omega, phi = sin_params
sin_wave = epsilon * np.sin(omega * (x - x_start) + phi)  # 注意这里平移了一下正弦波
y_left = (np.abs(x_left - x_offset) / (4 * focal_length)) ** (1 / power) + y_offset
y_right = np.flip(y_left)  # 得到开口向左的上半部分
y1 = np.concatenate([y_left, y_right]) + sin_wave
y2 = 2 * y_offset - y1  # 得到整个纺锤形的下半部分
```

$x_{start}$和$x_{end}$与前一种Fusiform相同