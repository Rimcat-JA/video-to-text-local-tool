"""時間軸ユーティリティ.

設計 4.1: 内部比較は整数マイクロ秒、文書表示はミリ秒単位。
時刻は「PTS x time_base - t0」から求める。可変フレームレートでも
「フレーム番号 / 公称FPS」を時刻として使わない。
"""

from __future__ import annotations

from fractions import Fraction

US = 1_000_000


def pts_to_us(pts: int, time_base: Fraction | tuple[int, int], t0_us: int = 0) -> int:
    """PTS と time_base から、t0 を原点とする整数マイクロ秒を求める。"""
    if isinstance(time_base, tuple):
        time_base = Fraction(time_base[0], time_base[1])
    exact = Fraction(pts) * Fraction(time_base) * US
    # 四捨五入は最近接偶数ではなく通常の丸めにして、往復変換のずれを避ける。
    return int(exact + Fraction(1, 2)) - t0_us if exact >= 0 else -int(-exact + Fraction(1, 2)) - t0_us


def seconds_to_us(seconds: float) -> int:
    return int(round(seconds * US))


def us_to_seconds(us: int) -> float:
    return us / US


def format_timestamp(us: int, *, ms: bool = True) -> str:
    """"H:MM:SS.mmm" 形式。負値は先頭に - を付ける。"""
    sign = "-" if us < 0 else ""
    us = abs(int(us))
    total_ms = us // 1000
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if ms:
        return f"{sign}{hours}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{sign}{hours}:{minutes:02d}:{seconds:02d}"


def format_srt_timestamp(us: int) -> str:
    """SRT の "HH:MM:SS,mmm"。SRT は負時刻を表現できないため 0 で下限を切る。"""
    us = max(0, int(us))
    total_ms = us // 1000
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def overlap_us(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """設計 8.1: overlap(B, U) = max(0, min(B.end, U.end) - max(B.start, U.start))"""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def contains_half_open(start: int, end: int, t: int) -> bool:
    """半開区間 [start, end) に t が入るか。設計 4.2。"""
    return start <= t < end
