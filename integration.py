"""
数值积分方法模块
包含：梯形法、辛普生第一法、辛普生第二法、乞贝雪夫法
以及端点坐标修正和增加半站法的实现。
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


# ============================================================
#  基础积分方法
# ============================================================

def trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """
    梯形法 (Trapezoidal Rule)
    ∫ y dx ≈ Σ[(y_i + y_{i+1})/2 * (x_{i+1} - x_i)]

    参数:
        x: 横坐标数组（站号）
        y: 纵坐标数组（半宽值，None已替换为0）
    返回: 积分值
    """
    if len(x) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        total += (y[i] + y[i + 1]) / 2.0 * dx
    return total


def trapezoidal_with_weights(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    梯形法 - 返回积分值和各项分量（用于计算表格展示）
    返回: (总积分, 各段积分分量数组)
    """
    if len(x) < 2:
        return 0.0, np.array([])
    segments = np.zeros(len(x) - 1)
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        segments[i] = (y[i] + y[i + 1]) / 2.0 * dx
        total += segments[i]
    return total, segments


def simpson_13(x: np.ndarray, y: np.ndarray) -> float:
    """
    辛普生第一法 (Simpson's 1/3 Rule)
    要求: 等间距，区间数为偶数
    ∫ y dx ≈ h/3 * [y0 + yn + 4(y1+y3+...) + 2(y2+y4+...)]

    对于不等间距或奇数区间，使用复合辛普生法（混合1/3和3/8法则）
    """
    n = len(x) - 1
    if n < 1:
        return 0.0
    if n == 1:
        return trapezoidal(x, y)

    # 检查是否为等间距
    dx = np.diff(x)
    if np.allclose(dx, dx[0], rtol=1e-8):
        h = dx[0]
        if n % 2 == 0:
            # 标准辛普生1/3法
            total = y[0] + y[-1]
            for i in range(1, n, 2):
                total += 4.0 * y[i]
            for i in range(2, n - 1, 2):
                total += 2.0 * y[i]
            return total * h / 3.0
        else:
            # 奇数区间：前n-3个区间用1/3法，最后3个区间用3/8法
            if n >= 3:
                total = simpson_13(x[:n-2], y[:n-1])
                # 最后3个区间用3/8法
                total += 3.0 * h / 8.0 * (y[n-3] + 3.0*y[n-2] + 3.0*y[n-1] + y[n])
                return total
            else:
                return trapezoidal(x, y)
    else:
        # 非等间距：逐段梯形法
        return trapezoidal(x, y)


def simpson_13_weights(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    辛普生第一法 - 返回积分值、各站乘数（权重系数*dx）和站距h
    用于生成计算表格
    返回: (总积分, 权重数组(对应各y值), 站距h)
    """
    n = len(x) - 1
    if n < 1:
        return 0.0, np.array([]), 0.0

    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1e-8):
        # 非等间距，回退到梯形法
        total, segs = trapezoidal_with_weights(x, y)
        return total, np.ones(len(y)), dx[0] if len(dx) > 0 else 1.0

    h = dx[0]
    weights = np.zeros(len(y))

    if n % 2 == 0:
        # 标准辛普生乘数: 1, 4, 2, 4, 2, ..., 4, 1
        weights[0] = 1.0
        weights[-1] = 1.0
        for i in range(1, n):
            weights[i] = 4.0 if i % 2 == 1 else 2.0
        total = np.sum(weights * y) * h / 3.0
    else:
        # 混合法
        # 先用1/3法处理前n-3个区间
        weights[:n-2] = 0
        if n - 2 >= 0:
            # 实际上是前n-3个区间用1/3法
            w_13 = np.zeros(n - 2 + 1)  # n-2+1 = n-1 个点
            w_13[0] = 1.0
            w_13[-1] = 1.0
            for i in range(1, n - 2):
                w_13[i] = 4.0 if i % 2 == 1 else 2.0
            weights[:n-1] += w_13
        # 最后3个区间用3/8法
        weights[n-3] += 1.0
        weights[n-2] += 3.0
        weights[n-1] += 3.0
        weights[n] += 1.0
        total = np.sum(weights * y) * h / 3.0 if n >= 3 else trapezoidal(x, y)

    return total, weights, h


def simpson_38(x: np.ndarray, y: np.ndarray) -> float:
    """
    辛普生第二法 (Simpson's 3/8 Rule)
    要求: 等间距，区间数为3的倍数
    ∫ y dx ≈ 3h/8 * [y0 + yn + 3(y1+y2+y4+y5+...) + 2(y3+y6+...)]
    """
    n = len(x) - 1
    if n < 1:
        return 0.0

    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1e-8):
        return trapezoidal(x, y)
    h = dx[0]

    if n % 3 == 0:
        # 标准辛普生3/8法
        total = y[0] + y[-1]
        for i in range(1, n):
            if i % 3 == 0:
                total += 2.0 * y[i]
            else:
                total += 3.0 * y[i]
        return total * 3.0 * h / 8.0
    else:
        # 区间数不是3的倍数
        # 先尽可能用3/8法，余下的用1/3法或梯形法
        total = 0.0
        m = (n // 3) * 3  # 可被3整除的最大区间数
        if m >= 3:
            # 前m个区间用3/8法
            nn = m
            sub_total = y[0] + y[m]
            for i in range(1, nn):
                sub_total += (2.0 if i % 3 == 0 else 3.0) * y[i]
            total += sub_total * 3.0 * h / 8.0
            # 余下部分
            remaining_n = n - m
            x_rem = np.arange(remaining_n + 1) * h
            total += simpson_13(x_rem, y[m:])
        else:
            total = simpson_13(x, y)
        return total


def simpson_38_weights(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    辛普生第二法 - 返回权重数组用于计算表格
    """
    n = len(x) - 1
    if n < 1:
        return 0.0, np.array([]), 0.0

    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1e-8):
        total = trapezoidal(x, y)
        return total, np.ones(len(y)), dx[0] if len(dx) > 0 else 1.0
    h = dx[0]

    weights = np.zeros(len(y))

    if n % 3 == 0:
        weights[0] = 1.0
        weights[-1] = 1.0
        for i in range(1, n):
            weights[i] = 2.0 if i % 3 == 0 else 3.0
        total = np.sum(weights * y) * 3.0 * h / 8.0
    else:
        m = (n // 3) * 3
        if m >= 3:
            weights[0] = 1.0
            weights[m] = 1.0
            for i in range(1, m):
                weights[i] = 2.0 if i % 3 == 0 else 3.0
            total = np.sum(weights[:m+1] * y[:m+1]) * 3.0 * h / 8.0
            x_rem = np.arange(n - m + 1) * h
            t2, w2, _ = simpson_13_weights(x_rem, y[m:])
            total += t2
        else:
            return simpson_13_weights(x, y)

    return total, weights, h


def chebyshev(x_range: Tuple[float, float], f: Callable[[float], float], n: int) -> float:
    """
    乞贝雪夫法 (Chebyshev's Rule)
    使用契比雪夫多项式零点作为积分节点。

    ∫[a,b] f(x)dx ≈ (b-a)/n * Σ f(x_k)
    其中 x_k = (a+b)/2 + (b-a)/2 * t_k
    t_k 是契比雪夫多项式的零点

    参数:
        x_range: (a, b) 积分区间
        f: 被积函数
        n: 节点数 (常用: 2, 3, 4, 5, 6, 7, 9)
    """
    a, b = x_range

    # 契比雪夫节点（在[-1, 1]上的零点）
    chebyshev_nodes = {
        2: [-0.5773502692, 0.5773502692],
        3: [-0.7071067812, 0.0, 0.7071067812],
        4: [-0.7946544723, -0.1875924741, 0.1875924741, 0.7946544723],
        5: [-0.8324974870, -0.3745414096, 0.0, 0.3745414096, 0.8324974870],
        6: [-0.8662468181, -0.4225186538, -0.2666354015, 0.2666354015, 0.4225186538, 0.8662468181],
        7: [-0.8838617008, -0.5296567753, -0.3239186261, 0.0, 0.3239186261, 0.5296567753, 0.8838617008],
        9: [-0.9115893077, -0.6010186554, -0.5287617831, -0.1679062469, 0.0,
            0.1679062469, 0.5287617831, 0.6010186554, 0.9115893077],
    }

    if n not in chebyshev_nodes:
        raise ValueError(f"乞贝雪夫法支持的节点数: {list(chebyshev_nodes.keys())}")

    nodes = chebyshev_nodes[n]
    total = 0.0
    for t in nodes:
        xk = (a + b) / 2.0 + (b - a) / 2.0 * t
        total += f(xk)

    return (b - a) / n * total


def chebyshev_discrete(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    乞贝雪夫法应用于离散数据点
    先在契比雪夫节点位置插值得到对应的y值，然后积分。

    参数:
        x, y: 已知数据点
        n: 乞贝雪夫节点数
    返回: (积分值, 契比雪夫节点x坐标, 插值得到的y值)
    """
    a, b = x[0], x[-1]

    chebyshev_nodes = {
        2: [-0.5773502692, 0.5773502692],
        3: [-0.7071067812, 0.0, 0.7071067812],
        4: [-0.7946544723, -0.1875924741, 0.1875924741, 0.7946544723],
        5: [-0.8324974870, -0.3745414096, 0.0, 0.3745414096, 0.8324974870],
        6: [-0.8662468181, -0.4225186538, -0.2666354015, 0.2666354015, 0.4225186538, 0.8662468181],
        7: [-0.8838617008, -0.5296567753, -0.3239186261, 0.0, 0.3239186261, 0.5296567753, 0.8838617008],
        9: [-0.9115893077, -0.6010186554, -0.5287617831, -0.1679062469, 0.0,
            0.1679062469, 0.5287617831, 0.6010186554, 0.9115893077],
    }

    if n not in chebyshev_nodes:
        n = min(chebyshev_nodes.keys(), key=lambda k: abs(k - n))

    nodes = chebyshev_nodes[n]
    cheb_x = np.array([(a + b) / 2.0 + (b - a) / 2.0 * t for t in nodes])
    # 插值
    cheb_y = np.interp(cheb_x, x, y)
    total = (b - a) / n * np.sum(cheb_y)

    return total, cheb_x, cheb_y


# ============================================================
#  端点坐标修正
# ============================================================

def endpoint_correction_linear(x: np.ndarray, y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    直线端点修正法
    在水线首尾端点处，根据相邻两站的半宽值线性外推，
    确定水线与船体型线的实际交点位置。

    原理：当水线在某端不经过整数站时（即端点站半宽为0），
    根据前两个非零站的半宽值线性外插求出实际端点位置。

    返回: (修正后的x数组, 修正后的y数组)
    """
    y = y_raw.copy()
    x_new = x.copy().astype(float)

    n = len(x)

    # 前端修正：找到第一个非零值
    first_nonzero = None
    for i in range(n):
        if y[i] > 0:
            first_nonzero = i
            break

    if first_nonzero is not None and first_nonzero > 0:
        # 利用第first_nonzero和first_nonzero+1（若有）线性外推
        if first_nonzero + 1 < n and y[first_nonzero + 1] > 0:
            # 两点线性外推
            x1, x2 = x[first_nonzero], x[first_nonzero + 1]
            y1_val, y2_val = y[first_nonzero], y[first_nonzero + 1]
            # 找到y=0的位置: x0 = x1 - y1*(x2-x1)/(y2-y1)
            x0 = x1 - y1_val * (x2 - x1) / (y2_val - y1_val)
            # 将x0之前的站移除，在x0处插入端点
            if x0 < x[first_nonzero - 1]:
                x0 = x[first_nonzero - 1]  # 至少保留一个端点站
            x_new = np.concatenate([[x0], x[first_nonzero:]])
            y = np.concatenate([[0.0], y[first_nonzero:]])
        elif first_nonzero >= 1:
            # 单点外推（假设通过原点）
            x0 = x[first_nonzero - 1]
            x_new = np.concatenate([[x0], x[first_nonzero:]])
            y = np.concatenate([[0.0], y[first_nonzero:]])

    # 后端修正
    n2 = len(x_new)
    last_nonzero = None
    for i in range(n2 - 1, -1, -1):
        if y[i] > 0:
            last_nonzero = i
            break

    if last_nonzero is not None and last_nonzero < n2 - 1:
        if last_nonzero - 1 >= 0 and y[last_nonzero - 1] > 0:
            x1, x2 = x_new[last_nonzero - 1], x_new[last_nonzero]
            y1_val, y2_val = y[last_nonzero - 1], y[last_nonzero]
            xn = x2 + y2_val * (x2 - x1) / (y1_val - y2_val)
            if xn > x_new[last_nonzero + 1]:
                xn = x_new[last_nonzero + 1]
            x_new = np.concatenate([x_new[:last_nonzero + 1], [xn]])
            y = np.concatenate([y[:last_nonzero + 1], [0.0]])

    return x_new, y


def endpoint_correction_parabolic(x: np.ndarray, y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    抛物线端点修正法
    利用前三个非零站的半宽值拟合抛物线，求水线与船体的实际交点。

    抛物线 y = ax² + bx + c 通过三点，求y=0时的x值。
    """
    y = y_raw.copy()
    x_new = x.copy().astype(float)

    n = len(x)

    # 找出所有非零值索引
    nonzero_idx = [i for i in range(n) if y[i] > 0]

    if len(nonzero_idx) < 3:
        # 不足三点，回退到线性修正
        return endpoint_correction_linear(x, y_raw)

    # 前端修正
    first = nonzero_idx[0]
    if first > 0:
        # 取前三个非零点
        idx = nonzero_idx[:3]
        # 用三点拟合抛物线
        xi = x[idx]
        yi = y[idx]
        # 解方程组: yi = a*xi² + b*xi + c
        A = np.column_stack([xi**2, xi, np.ones(3)])
        try:
            a_coef, b_coef, c_coef = np.linalg.solve(A, yi)
            # 求 y=0 的根（靠内的那个根）
            disc = b_coef**2 - 4 * a_coef * c_coef
            if disc >= 0:
                roots = np.roots([a_coef, b_coef, c_coef])
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                if real_roots:
                    x0 = min(real_roots, key=lambda r: abs(r - x[first]))
                    if x0 < x[first] and x0 > x[first - 1]:
                        x_new = np.concatenate([[x0], x[first:]])
                        y = np.concatenate([[0.0], y[first:]])
        except np.linalg.LinAlgError:
            return endpoint_correction_linear(x, y_raw)

    # 后端修正
    n2 = len(y)
    nonzero_idx2 = [i for i in range(n2) if y[i] > 0]
    if len(nonzero_idx2) < 3:
        x2, y2 = endpoint_correction_linear(x_new[:len(y)], y)
        return x2, y2

    last = nonzero_idx2[-1]
    if last < n2 - 1:
        idx = nonzero_idx2[-3:]
        xi = x_new[idx]
        yi = y[idx]
        A = np.column_stack([xi**2, xi, np.ones(3)])
        try:
            a_coef, b_coef, c_coef = np.linalg.solve(A, yi)
            roots = np.roots([a_coef, b_coef, c_coef])
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            if real_roots:
                xn = max(real_roots, key=lambda r: abs(r - x_new[last]))
                if xn > x_new[last] and xn < x_new[last + 1]:
                    x_new = np.concatenate([x_new[:last + 1], [xn]])
                    y = np.concatenate([y[:last + 1], [0.0]])
        except np.linalg.LinAlgError:
            return endpoint_correction_linear(x_new[:len(y)], y)

    return x_new, y


# ============================================================
#  纵坐标计算辅助函数
# ============================================================

def compute_longitudinal_moment(x: np.ndarray, y: np.ndarray, x_ref: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    计算纵倾力矩臂和纵倾力矩相关分量
    x_ref: 参考点（通常取船中，x_ref = N/2）
    返回: (力矩臂数组, 对参考点的力矩)
    """
    arms = x - x_ref
    return arms, np.sum(arms * y)


# ============================================================
#  综合积分函数（用于垂直方向积分）
# ============================================================

def integrate_vertical(z: np.ndarray, values: np.ndarray, method: str = 'trapezoidal') -> float:
    """
    垂直方向（吃水方向）积分
    用于计算横剖面面积等
    """
    if method == 'trapezoidal':
        return trapezoidal(z, values)
    elif method == 'simpson13':
        return simpson_13(z, values)
    elif method == 'simpson38':
        return simpson_38(z, values)
    else:
        return trapezoidal(z, values)
