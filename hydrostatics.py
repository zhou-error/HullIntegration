"""
船舶静力学计算模块
计算水线面面积、漂心、排水体积、浮心、惯性矩等静水力要素。
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field

from ship_data import (
    ShipOffsetData, get_valid_station_range,
    interpolate_half_stations, fill_none_with_zero, get_waterline_breadths
)
from integration import (
    trapezoidal, simpson_13, simpson_38, chebyshev_discrete, chebyshev,
    endpoint_correction_linear, endpoint_correction_parabolic,
    simpson_13_weights, simpson_38_weights, trapezoidal_with_weights
)


@dataclass
class HydrostaticResult:
    """静水力计算结果"""
    method_name: str = ""
    dL: float = None  # 站距 (mm)

    # 各水线计算结果
    waterline_results: list = field(default_factory=list)

    # 排水体积和浮心（各吃水）
    displacement_results: list = field(default_factory=list)

    # 计算过程数据（用于生成报告表格）
    process_data: Dict = field(default_factory=dict)


@dataclass
class WaterlineResult:
    """单条水线的静水力要素"""
    wl_label: str = ""
    wl_height: float = 0.0  # mm

    # 水线面
    waterplane_area: float = 0.0  # mm² (需×dL)
    Aw_multiplier: str = ""  # 如 "2×dL×Σwy"

    # 漂心
    xf: float = 0.0  # 漂心距船中距离（站号单位）
    xf_moment: float = 0.0  # 对船中的面积矩

    # 惯性矩
    transverse_IT: float = 0.0  # 横向惯性矩 (mm⁴)
    longitudinal_IL: float = 0.0  # 纵向惯性矩 (mm⁵，需×dL³)

    # 水线面系数
    Cwp: float = 0.0


@dataclass
class DisplacementResult:
    """某吃水下的排水量和浮心"""
    draft: float = 0.0  # mm
    displacement_volume: float = 0.0  # mm³ (需×dL)
    displacement_mass: float = 0.0  # 排水量（吨），需知道船长

    LCB: float = 0.0  # 浮心距船中纵向位置（站号单位）
    KB: float = 0.0  # 浮心垂向高度 (mm)
    LCB_from_AP: float = 0.0  # 浮心距尾垂线距离（站号单位）

    CB: float = 0.0  # 方形系数
    CM: float = 0.0  # 中横剖面系数
    CWP: float = 0.0  # 水线面系数
    CP: float = 0.0  # 菱形系数


class HydrostaticCalculator:
    """静水力计算器"""

    def __init__(self, data: ShipOffsetData, dL: float = None):
        """
        参数:
            data: 型值表数据
            dL: 站距 (mm)，若为None则使用data中的值
        """
        self.data = data
        self.dL = dL or data.station_spacing

    def calc_waterplane_properties(self, wl_idx: int,
                                    integration_method: str = 'trapezoidal',
                                    use_endpoint_correction: bool = True,
                                    use_half_stations: bool = False,
                                    endpoint_type: str = 'linear') -> Tuple[Dict, Dict]:
        """
        计算某水线的水线面静水力要素。

        参数:
            wl_idx: 水线索引 (0=基线, 1=WL1, ...)
            integration_method: 'trapezoidal', 'simpson13', 'simpson38', 'chebyshev'
            use_endpoint_correction: 是否使用端点修正
            use_half_stations: 是否使用增加半站法
            endpoint_type: 端点修正类型 'linear' 或 'parabolic'

        返回: (结果字典, 过程数据字典)
        """
        # 获取原始数据
        x_orig, y_orig_raw = get_waterline_breadths(self.data, wl_idx)
        y_orig = fill_none_with_zero(y_orig_raw)

        # 端点修正
        x_ec = x_orig.copy()
        y_ec = y_orig.copy()
        endpoint_info = {}

        if use_endpoint_correction:
            # 检查是否需要端点修正（水线是否在端点处为零）
            nonzero_idx = np.where(y_orig > 0)[0]
            if len(nonzero_idx) > 0:
                needs_correction = (nonzero_idx[0] > 0) or (nonzero_idx[-1] < len(y_orig) - 1)
                if needs_correction:
                    if endpoint_type == 'parabolic':
                        x_ec, y_ec = endpoint_correction_parabolic(x_orig, y_orig)
                    else:
                        x_ec, y_ec = endpoint_correction_linear(x_orig, y_orig)
                    endpoint_info['corrected'] = True
                    endpoint_info['x_range'] = (x_ec[0], x_ec[-1])
                    endpoint_info['n_points'] = len(x_ec)
                else:
                    endpoint_info['corrected'] = False

        # 增加半站
        if use_half_stations:
            # 在半站位置插值
            x_expanded = []
            y_expanded = []
            for i in range(len(x_ec) - 1):
                x_expanded.append(x_ec[i])
                y_expanded.append(y_ec[i])
                x_mid = (x_ec[i] + x_ec[i + 1]) / 2.0
                y_mid = (y_ec[i] + y_ec[i + 1]) / 2.0
                x_expanded.append(x_mid)
                y_expanded.append(y_mid)
            x_expanded.append(x_ec[-1])
            y_expanded.append(y_ec[-1])
            x_use = np.array(x_expanded)
            y_use = np.array(y_expanded)
        else:
            x_use = x_ec
            y_use = y_ec

        process = {
            'x_orig': x_orig, 'y_orig': y_orig, 'y_orig_raw': y_orig_raw,
            'x_used': x_use, 'y_used': y_use,
            'x_ec': x_ec, 'y_ec': y_ec,
            'endpoint_info': endpoint_info,
            'use_half_stations': use_half_stations,
            'integration_method': integration_method,
        }

        # 根据积分方法计算
        if integration_method == 'chebyshev':
            # 找出有效范围
            valid_mask = y_orig > 0
            if np.sum(valid_mask) >= 3:
                x_valid = x_orig[valid_mask]
                y_valid = y_orig[valid_mask]
                # 使用6节点乞贝雪夫法
                area_half, cheb_x, cheb_y = chebyshev_discrete(x_valid, y_valid, n=9)
                process['cheb_x'] = cheb_x
                process['cheb_y'] = cheb_y
                Aw = 2.0 * area_half
            else:
                Aw = 0.0
                area_half = 0.0
        else:
            # 梯形法、辛普生法
            area_half = self._integrate_1d(x_use, y_use, integration_method)
            Aw = 2.0 * area_half

        # 漂心计算
        x_ref = self.data.num_stations / 2.0  # 船中(站号)
        if integration_method == 'chebyshev' and 'cheb_x' in process:
            # Chebyshev积分需使用完整公式: ∫(x-xref)y dx ≈ (b-a)/n * Σ(xk-xref)y(xk)
            cheb_x = process['cheb_x']
            cheb_y = process['cheb_y']
            a_c, b_c = cheb_x[0], cheb_x[-1]
            n_cheb = len(cheb_x)
            moment_x = (b_c - a_c) / n_cheb * np.sum(cheb_y * (cheb_x - x_ref))
        else:
            moment_x = self._integrate_1d(x_use, (x_use - x_ref) * y_use, integration_method)

        if area_half > 0:
            xf = moment_x / area_half  # 漂心距船中距离（站号单位）
        else:
            xf = 0.0

        # 横向惯性矩 IT = 2/3 ∫ y³ dx
        y_cubed = y_use ** 3
        IT_half = self._integrate_1d(x_use, y_cubed, integration_method)
        IT = 2.0 / 3.0 * IT_half

        # 纵向惯性矩 IL = 2 ∫ (x-xF)²y dx = 2[∫ x²y dx - 2xF ∫ xy dx + xF² ∫ y dx]
        # 对船中的惯性矩
        x_dev = x_use - x_ref
        IL_half_midship = self._integrate_1d(x_use, x_dev**2 * y_use, integration_method)
        if area_half > 0:
            IL_half = IL_half_midship - area_half * xf**2  # 对漂心轴的惯性矩
        else:
            IL_half = 0.0
        IL = 2.0 * IL_half

        # 水线面系数 Cwp = Aw / (L × Bmax)
        B_max = 2.0 * np.max(y_use)
        L_stations = x_use[-1] - x_use[0]  # 以站号为单位
        if L_stations > 0 and B_max > 0:
            Cwp = area_half / (L_stations * B_max / 2.0)
        else:
            Cwp = 0.0

        result = {
            'wl_idx': wl_idx,
            'waterplane_area_half': area_half,  # ∫ydx（半面积）
            'waterplane_area': Aw,  # 2∫ydx
            'xf': xf,  # 漂心距船中（站号单位）
            'moment_x_half': moment_x,  # ∫(x-x_ref)y dx
            'IT_half': IT_half,  # ∫y³dx
            'IT': IT,  # 2/3∫y³dx
            'IL_half_midship': IL_half_midship,  # ∫(x-x_ref)²y dx
            'IL_half': IL_half,  # 对漂心轴的半纵向惯性矩
            'IL': IL,  # 对漂心轴的全纵向惯性矩
            'Cwp': Cwp,
            'max_breadth': 2.0 * np.max(y_use),
            'L_waterline_stations': L_stations,
        }

        return result, process

    def _integrate_1d(self, x: np.ndarray, y: np.ndarray, method: str) -> float:
        """根据指定方法进行一维积分"""
        if method == 'trapezoidal':
            return trapezoidal(x, y)
        elif method == 'simpson13':
            return simpson_13(x, y)
        elif method == 'simpson38':
            return simpson_38(x, y)
        elif method == 'chebyshev':
            # 找非零点(绝对值>0)用切比雪夫法
            nonzero = np.abs(y) > 0
            if np.sum(nonzero) >= 3:
                val, _, _ = chebyshev_discrete(x[nonzero], y[nonzero], n=9)
                return val
            else:
                return trapezoidal(x, y)
        else:
            return trapezoidal(x, y)

    def calc_sectional_area(self, station: float, draft: float,
                             method: str = 'trapezoidal') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        计算某站横剖面面积（半面积 ∫₀ᵀ y dz）
        参数:
            station: 站号
            draft: 计算吃水 (mm)
            method: 垂直方向积分方法
        返回: (半剖面面积, 吃水z数组, 半宽y数组)
        """
        z_vals = [0.0]
        y_vals = []

        # 基线 (z=0)
        hb_base = self.data.get_half_breadth(station, 0)
        if hb_base is not None:
            y_vals.append(float(hb_base))
        else:
            # 基线处无线型，取0或第一个有效水线的插值
            # 使用第一个有效水线值的一半（近似锥形底）
            found = False
            for wl_idx in range(1, self.data.num_waterlines):
                hb = self.data.get_half_breadth(station, wl_idx)
                if hb is not None:
                    y_vals.append(float(hb) * 0.3)  # 近似
                    found = True
                    break
            if not found:
                y_vals.append(0.0)

        # 各水线，只到指定吃水
        for wl_idx in range(1, self.data.num_waterlines):
            z_wl = self.data.waterline_heights[wl_idx]
            if z_wl > draft + 1e-6:
                # 需要在当前水线和上一水线之间插值
                if len(z_vals) > 1:
                    z_prev = z_vals[-1]
                    y_prev = y_vals[-1]
                    # 线性插值到精确吃水
                    z_vals.append(float(draft))
                    # 获取本水线的半宽（即使超过draft也读取用于插值）
                    hb_cur = self.data.get_half_breadth(station, wl_idx)
                    y_cur = 0.0 if hb_cur is None else float(hb_cur)
                    if y_prev > 0 or y_cur > 0:
                        frac = (draft - z_prev) / (z_wl - z_prev)
                        y_draft = y_prev + frac * (y_cur - y_prev)
                        y_vals.append(max(0.0, y_draft))
                    else:
                        y_vals.append(0.0)
                break
            hb = self.data.get_half_breadth(station, wl_idx)
            z_vals.append(z_wl)
            y_vals.append(0.0 if hb is None else float(hb))

        z_arr = np.array(z_vals)
        y_arr = np.array(y_vals)

        # 积分求半剖面面积
        area_half = self._integrate_1d(z_arr, y_arr, method)

        return area_half, z_arr, y_arr

    def calc_displacement_properties(self, draft: float,
                                      integration_method: str = 'trapezoidal',
                                      use_endpoint_correction: bool = True,
                                      use_half_stations: bool = False,
                                      vertical_method: str = 'trapezoidal') -> Dict:
        """
        计算指定吃水下的排水体积和浮心。

        方法A: 先计算横剖面面积曲线，再沿船长积分
        1. 计算各站横剖面面积（截断到指定吃水）
        2. 沿船长积分求排水体积

        参数:
            draft: 吃水 (mm)

        注意：位移体积 = 2 * ∫ As_half(x) dx（x以站号为单位），
        结果需乘以dL（站距mm）才是实际mm³值。
        船型系数CB/CM/CP/CWP是无量纲量，不依赖dL。
        """
        N = self.data.num_stations  # 20

        # 计算各站的横剖面面积（直到指定吃水）
        station_areas = {}
        station_beams = {}  # 各站在该吃水下的半宽
        for s in self.data.stations:
            if s != int(s) and s < 0:
                continue
            area_half, z_arr, y_arr = self.calc_sectional_area(s, draft, vertical_method)
            station_areas[s] = area_half
            station_beams[s] = float(np.max(y_arr)) if len(y_arr) > 0 else 0.0

        # 只使用整数站（0, 1, ..., 20）
        int_stations = np.array([s for s in self.data.stations if s == int(s) and s >= 0])
        areas = np.array([station_areas[s] for s in int_stations])

        # 沿船长积分求排水体积
        x_orig = int_stations.copy()
        y_orig = areas.copy()

        if use_endpoint_correction:
            nonzero_idx = np.where(y_orig > 0)[0]
            if len(nonzero_idx) > 0:
                needs_correction = (nonzero_idx[0] > 0) or (nonzero_idx[-1] < len(y_orig) - 1)
                if needs_correction:
                    x_ec, y_ec = endpoint_correction_linear(x_orig, y_orig)
                else:
                    x_ec, y_ec = x_orig, y_orig
            else:
                x_ec, y_ec = x_orig, y_orig
        else:
            x_ec, y_ec = x_orig, y_orig

        # 半站
        if use_half_stations:
            x_use = []
            y_use = []
            for i in range(len(x_ec) - 1):
                x_use.append(x_ec[i])
                y_use.append(y_ec[i])
                x_use.append((x_ec[i] + x_ec[i + 1]) / 2.0)
                y_use.append((y_ec[i] + y_ec[i + 1]) / 2.0)
            x_use.append(x_ec[-1])
            y_use.append(y_ec[-1])
            x_use = np.array(x_use)
            y_use = np.array(y_use)
        else:
            x_use, y_use = x_ec, y_ec

        # 排水半体积 ∫ As_half dx (x以站号为单位)
        vol_half = self._integrate_1d(x_use, y_use, integration_method)
        displacement = 2.0 * vol_half  # 需×dL才得实际mm³

        # 浮心纵向坐标 LCB（站号单位，距船中）
        x_ref = N / 2.0  # 船中站号=10
        moment_half = self._integrate_1d(x_use, (x_use - x_ref) * y_use, integration_method)
        if vol_half > 0:
            LCB = moment_half / vol_half
        else:
            LCB = 0.0

        # 浮心垂向坐标 KB
        # KB = ∫₀ᵀ z·Aw(z) dz / ∫₀ᵀ Aw(z) dz（梯形积分）
        sum_Aw = 0.0
        sum_zAw = 0.0
        prev_z = 0.0
        prev_Aw = 0.0

        for wi in range(1, self.data.num_waterlines):
            z_wl = self.data.waterline_heights[wi]
            if z_wl > draft + 1e-6:
                # 在draft处插值
                if prev_z < draft:
                    r_prev, _ = self.calc_waterplane_properties(
                        wi - 1, integration_method, use_endpoint_correction,
                        use_half_stations, 'linear'
                    )
                    r_cur, _ = self.calc_waterplane_properties(
                        wi, integration_method, use_endpoint_correction,
                        use_half_stations, 'linear'
                    )
                    frac = (draft - prev_z) / (z_wl - prev_z)
                    Aw_draft_prev = r_prev['waterplane_area']
                    Aw_draft_cur = r_cur['waterplane_area']
                    Aw_draft = Aw_draft_prev + frac * (Aw_draft_cur - Aw_draft_prev)
                    sum_Aw += Aw_draft
                    sum_zAw += draft * Aw_draft
                break

            r, _ = self.calc_waterplane_properties(
                wi, integration_method, use_endpoint_correction,
                use_half_stations, 'linear'
            )
            Aw = r['waterplane_area']
            sum_Aw += Aw
            sum_zAw += z_wl * Aw
            prev_z = z_wl
            prev_Aw = Aw

        if sum_Aw > 0:
            KB = sum_zAw / sum_Aw
        else:
            KB = 0.0

        # --- 船型系数 ---
        # B: 该吃水下的最大水线宽（取船中站9、10的最大半宽）
        B_half = 0.0
        for s in [9, 10, 8, 11]:
            if s in station_beams:
                B_half = max(B_half, station_beams[s])
        B = 2.0 * B_half

        # Am: 中横剖面面积（取站9、10的最大值）
        mid_area_half = 0.0
        for s in [9, 10, 8, 11]:
            if s in station_areas:
                mid_area_half = max(mid_area_half, station_areas[s])
        Am = 2.0 * mid_area_half

        L_stations = x_use[-1] - x_use[0]

        # CB = ∇ / (L * B * T)
        # ∇ = 2 * vol_half * dL (mm³)
        # L = L_stations * dL (mm)
        # CB = (2*vol_half*dL) / (L_stations*dL * B * T) = displacement / (L_stations * B * T)
        # displacement = 2*vol_half
        CB = displacement / (L_stations * B * draft) if (L_stations * B * draft) > 0 else 0.0

        # CM = Am / (B * T)
        CM = Am / (B * draft) if (B * draft) > 0 else 0.0

        # CP = CB / CM = ∇ / (Am * L)
        if CM > 0.001:
            CP = CB / CM
        else:
            CP = displacement / (Am * L_stations) if (Am * L_stations) > 0 else 0.0

        # CWP = Aw / (L * B) at the given draft
        if prev_Aw > 0:
            # 使用上一水线的CWP
            CWP = prev_Aw / (L_stations * B) if (L_stations * B) > 0 else 0.0
        else:
            CWP = 0.0

        return {
            'draft': draft,
            'displacement_half': vol_half,
            'displacement_volume': displacement,
            'LCB': LCB,
            'LCB_from_AP': x_ref + LCB,
            'KB': KB,
            'CB': CB,
            'CM': CM,
            'CWP': CWP,
            'CP': CP,
            'Am': Am,
            'B': B,
            'L_stations': L_stations,
            'x_used': x_use, 'y_used': y_use,
            'int_stations': int_stations, 'station_areas': station_areas,
        }

    def calc_waterplane_table_data(self, wl_idx: int, method: str,
                                    use_ec: bool, use_hs: bool) -> Dict:
        """
        获取用于生成计算表格的数据（梯形法或辛普生法）。
        返回包含各站数据的字典。
        """
        x_orig, y_orig_raw = get_waterline_breadths(self.data, wl_idx)
        y_orig = fill_none_with_zero(y_orig_raw)
        n_stations = len(x_orig)

        # 端点修正
        if use_ec:
            x_ec, y_ec = endpoint_correction_linear(x_orig, y_orig)
        else:
            x_ec, y_ec = x_orig.copy(), y_orig.copy()

        # 半站
        if use_hs:
            x_use = []
            y_use = []
            for i in range(len(x_ec) - 1):
                x_use.append(x_ec[i])
                y_use.append(y_ec[i])
                x_use.append((x_ec[i] + x_ec[i + 1]) / 2.0)
                y_use.append((y_ec[i] + y_ec[i + 1]) / 2.0)
            x_use.append(x_ec[-1])
            y_use.append(y_ec[-1])
            x_use = np.array(x_use)
            y_use = np.array(y_use)
        else:
            x_use, y_use = x_ec, y_ec

        table_data = {
            'station': x_use,
            'breadth': y_use,
            'breadth_orig': y_orig,
            'station_orig': x_orig,
        }

        # 根据方法计算权重和乘数
        if method == 'trapezoidal':
            total, segments = trapezoidal_with_weights(x_use, y_use)
            table_data['total'] = total
            table_data['segments'] = segments
            table_data['method_label'] = '梯形法'
        elif method == 'simpson13':
            total, weights, h_val = simpson_13_weights(x_use, y_use)
            table_data['total'] = total
            table_data['weights'] = weights
            table_data['h'] = h_val
            table_data['method_label'] = '辛普生第一法'
            # 加权值
            table_data['weighted'] = weights * y_use
        elif method == 'simpson38':
            total, weights, h_val = simpson_38_weights(x_use, y_use)
            table_data['total'] = total
            table_data['weights'] = weights
            table_data['h'] = h_val
            table_data['method_label'] = '辛普生第二法'
            table_data['weighted'] = weights * y_use

        return table_data
