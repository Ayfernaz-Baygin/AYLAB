"""AYLAB görsel kimliği: marka, renk sistemi, tipografi ve stil sayfası.

Renk sistemi
------------
* Zemin: derin lacivert → grafit → antrasit katmanları (saf siyah yok).
* Vurgu: tek bir sabit vurgu rengi yerine, seçili kategorinin *dalga boyu*.
  Yedi kategori görünür spektrumu (680 → 415 nm) paylaşır; arayüz, seçilen
  kategoriye "ayarlanır" — slider tutamakları, odak çerçeveleri, Apply düğmesi
  ve tayf çizgisindeki tepe noktası o rengi alır.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QLinearGradient, QPainterPath

# --------------------------------------------------------------------------
# Marka — arayüzdeki bütün marka yazıları buradan gelir
# --------------------------------------------------------------------------

BRAND_NAME = "AYLAB"
BRAND_DESCRIPTOR = "Image Processing Laboratory"
BRAND_TAGLINE = "BEYOND THE PIXEL."
BRAND_TITLE = f"{BRAND_NAME} — {BRAND_DESCRIPTOR}"

# --------------------------------------------------------------------------
# Renkler
# --------------------------------------------------------------------------

BG_DEEP = "#0A0F1B"  # pencere zemini — derin lacivert
BG_STAGE = "#0C1322"  # görüntü sahnesi
PANEL = "#111929"  # grafit-lacivert panel
PANEL_RAISED = "#162034"
ANTHRACITE = "#1B2232"
LINE = "#243049"
LINE_SOFT = "#19223A"
TEXT = "#DCE3EF"
TEXT_DIM = "#8693AC"
TEXT_FAINT = "#56627B"

OK = "#6BD98B"
WARN = "#F2C14E"
ERROR = "#FF6B5E"

# Kategorilerin dalga boylarına karşılık gelen, koyu zeminde dengelenmiş tonlar.
SPECTRAL_STOPS: tuple[tuple[int, str], ...] = (
    (415, "#9B7BFF"),
    (455, "#5B8DFF"),
    (495, "#3FD0D4"),
    (535, "#6BD98B"),
    (580, "#F2C14E"),
    (625, "#FF8A3D"),
    (680, "#FF5E57"),
)
SPECTRUM_MIN_NM = 400
SPECTRUM_MAX_NM = 700
NEUTRAL_ACCENT = "#7FA7D9"  # Hiçbir kategori seçili değilken


def wavelength_color(nm: float) -> QColor:
    """Dalga boyunu AYLAB tayf paletinde bir renge çevirir (doğrusal ara değer)."""
    stops = SPECTRAL_STOPS
    if nm <= stops[0][0]:
        return QColor(stops[0][1])
    if nm >= stops[-1][0]:
        return QColor(stops[-1][1])
    for (n0, c0), (n1, c1) in zip(stops, stops[1:]):
        if n0 <= nm <= n1:
            t = (nm - n0) / (n1 - n0)
            a, b = QColor(c0), QColor(c1)
            return QColor(
                round(a.red() + (b.red() - a.red()) * t),
                round(a.green() + (b.green() - a.green()) * t),
                round(a.blue() + (b.blue() - a.blue()) * t),
            )
    return QColor(NEUTRAL_ACCENT)  # pragma: no cover


def wavelength_to_unit(nm: float) -> float:
    """Dalga boyunu tayf şeridi üzerinde 0..1 konuma çevirir (mor solda)."""
    return max(0.0, min(1.0, (nm - SPECTRUM_MIN_NM) / (SPECTRUM_MAX_NM - SPECTRUM_MIN_NM)))


def spectrum_gradient(x1: float, y1: float, x2: float, y2: float, alpha: int = 255) -> QLinearGradient:
    grad = QLinearGradient(QPointF(x1, y1), QPointF(x2, y2))
    for nm, hex_color in SPECTRAL_STOPS:
        c = QColor(hex_color)
        c.setAlpha(alpha)
        grad.setColorAt(wavelength_to_unit(nm), c)
    return grad


def with_alpha(color: QColor | str, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def chamfer_path(x: float, y: float, w: float, h: float, cut: float = 10.0,
                 corners: str = "tl,br") -> QPainterPath:
    """Köşeleri kesik (pah kırılmış) dikdörtgen yolu. corners: tl,tr,br,bl."""
    c = {k: (cut if k in corners else 0.0) for k in ("tl", "tr", "br", "bl")}
    path = QPainterPath()
    path.moveTo(x + c["tl"], y)
    path.lineTo(x + w - c["tr"], y)
    path.lineTo(x + w, y + c["tr"])
    path.lineTo(x + w, y + h - c["br"])
    path.lineTo(x + w - c["br"], y + h)
    path.lineTo(x + c["bl"], y + h)
    path.lineTo(x, y + h - c["bl"])
    path.lineTo(x, y + c["tl"])
    path.closeSubpath()
    return path


# --------------------------------------------------------------------------
# Tipografi
# --------------------------------------------------------------------------

_DISPLAY_CANDIDATES = ("Bahnschrift", "DIN Alternate", "Segoe UI Variable Display", "Segoe UI")
_BODY_CANDIDATES = ("Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial")
_MONO_CANDIDATES = ("Cascadia Mono", "Cascadia Code", "JetBrains Mono", "Consolas", "Menlo", "Courier New")

_family_cache: dict[str, str] = {}


def _pick(role: str, candidates: tuple[str, ...]) -> str:
    if role not in _family_cache:
        available = set(QFontDatabase.families())
        _family_cache[role] = next((f for f in candidates if f in available), "")
    return _family_cache[role]


def display_font(size: float, weight: QFont.Weight = QFont.Weight.DemiBold, spacing: float = 0.0) -> QFont:
    """Başlıklar ve etiketler: mühendislik çizimi hissi veren DIN tarzı yüz."""
    f = QFont(_pick("display", _DISPLAY_CANDIDATES) or QFont().family())
    f.setPointSizeF(size)
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def body_font(size: float = 9.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont(_pick("body", _BODY_CANDIDATES) or QFont().family())
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def mono_font(size: float = 9.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Sayısal okumalar: ölçüm cihazı ekranı gibi eş aralıklı."""
    family = _pick("mono", _MONO_CANDIDATES)
    f = QFont(family) if family else QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSizeF(size)
    f.setWeight(weight)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def font_css(role: str) -> str:
    families = {"display": _DISPLAY_CANDIDATES, "body": _BODY_CANDIDATES, "mono": _MONO_CANDIDATES}[role]
    picked = _pick(role, families)
    return f'"{picked}"' if picked else "sans-serif"


# --------------------------------------------------------------------------
# Stil sayfası
# --------------------------------------------------------------------------


def build_stylesheet(accent: QColor | str = NEUTRAL_ACCENT) -> str:
    a = QColor(accent)
    acc = a.name()
    acc_soft = f"rgba({a.red()},{a.green()},{a.blue()},40)"
    acc_mid = f"rgba({a.red()},{a.green()},{a.blue()},110)"
    body, mono, display = font_css("body"), font_css("mono"), font_css("display")
    return f"""
QMainWindow, QDialog {{ background: {BG_DEEP}; }}
QWidget {{ color: {TEXT}; font-family: {body}; font-size: 9.5pt; }}
QToolTip {{
    background: {PANEL_RAISED}; color: {TEXT}; border: 1px solid {LINE};
    padding: 5px 8px; font-family: {mono}; font-size: 8.5pt;
}}

/* --- Kaydırma alanları ve çubuklar --- */
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {acc_mid}; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {LINE_SOFT}; }}
QSplitter::handle:hover {{ background: {acc_mid}; }}

/* --- Sayısal girişler --- */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_DEEP}; border: 1px solid {LINE}; border-radius: 0;
    padding: 3px 6px; font-family: {mono}; font-size: 9pt; color: {TEXT};
    selection-background-color: {acc_mid};
}}
/* Fusion + QSS'te gizli artır/azalt düğmeleri metin alanını 1 px'e sıkıştırır;
   kökeni kenarlığa almak alanın tüm genişliği kullanmasını sağlar. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 0px; border: none;
}}
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {TEXT_FAINT}; }}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {acc}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM}; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_RAISED}; border: 1px solid {LINE}; outline: 0;
    selection-background-color: {acc_soft}; selection-color: {TEXT};
    font-family: {mono};
}}

/* --- Kaydırıcı: ince ray + dikey "okuma çizgisi" tutamak --- */
QSlider {{ min-height: 20px; }}
QSlider::groove:horizontal {{ height: 2px; background: {LINE}; }}
QSlider::sub-page:horizontal {{ background: {acc}; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 4px; margin: -8px 0; border: none;
}}
QSlider::handle:horizontal:hover {{ background: {acc}; width: 6px; }}
QSlider::handle:horizontal:disabled {{ background: {TEXT_FAINT}; }}
QSlider::sub-page:horizontal:disabled {{ background: {TEXT_FAINT}; }}

/* --- Segmentli seçim --- */
QPushButton#Segment {{
    background: transparent; border: 1px solid {LINE}; border-radius: 0;
    color: {TEXT_DIM}; padding: 4px 8px;
    font-family: {display}; font-size: 8.5pt; letter-spacing: 0.5px;
}}
QPushButton#Segment:hover {{ color: {TEXT}; border-color: {TEXT_FAINT}; }}
QPushButton#Segment:checked {{ color: {acc}; background: {acc_soft}; border-color: {acc}; }}

/* --- Ana eylem ve ikincil düğmeler --- */
QPushButton#PrimaryAction {{
    background: {acc}; color: {BG_DEEP}; border: none; border-radius: 0;
    padding: 8px 14px; font-family: {display}; font-size: 10pt; font-weight: 600;
    letter-spacing: 1.5px; text-align: left;
}}
QPushButton#PrimaryAction:hover {{ background: {a.lighter(115).name()}; }}
QPushButton#PrimaryAction:pressed {{ background: {a.darker(115).name()}; }}
QPushButton#PrimaryAction:disabled {{ background: {ANTHRACITE}; color: {TEXT_FAINT}; }}
QPushButton#GhostAction {{
    background: transparent; color: {TEXT_DIM}; border: 1px solid {LINE}; border-radius: 0;
    padding: 7px 10px; font-family: {display}; font-size: 9pt; letter-spacing: 1px;
}}
QPushButton#GhostAction:hover {{ color: {TEXT}; border-color: {TEXT_FAINT}; }}
QPushButton#GhostAction:disabled {{ color: {TEXT_FAINT}; border-color: {LINE_SOFT}; }}

/* --- Sahne araç şeridi --- */
QToolButton#StageTool {{
    background: transparent; border: 1px solid transparent; color: {TEXT_DIM};
    padding: 3px 8px; font-family: {display}; font-size: 8.5pt; letter-spacing: 1px;
}}
QToolButton#StageTool:hover {{ color: {TEXT}; border-color: {LINE}; }}
QToolButton#StageTool:checked {{ color: {acc}; border-color: {acc}; background: {acc_soft}; }}
QToolButton#StageTool:disabled {{ color: {TEXT_FAINT}; }}

/* --- Durum çubuğu --- */
QStatusBar {{
    background: {PANEL}; border-top: 1px solid {LINE}; color: {TEXT_DIM};
    font-family: {mono}; font-size: 8.5pt; min-height: 26px;
}}
QStatusBar::item {{ border: none; }}

QLabel#Eyebrow {{ color: {acc}; font-family: {mono}; font-size: 8pt; letter-spacing: 1px; }}
QLabel#SectionTitle {{ color: {TEXT_DIM}; font-family: {display}; font-size: 8.5pt; letter-spacing: 2px; }}
QLabel#Muted {{ color: {TEXT_DIM}; }}
QLabel#Faint {{ color: {TEXT_FAINT}; font-family: {mono}; font-size: 8pt; }}
QLabel#ParamLabel {{ color: {TEXT_DIM}; font-family: {display}; font-size: 8.5pt; letter-spacing: 1px; }}
"""
