"""Ders kapsamındaki kategori ve algoritma tanımları.

Burada yalnızca *tanımlar* (ad, açıklama, parametreler) vardır. İşlevler
ilgili modüllerde (filters.py, morphology.py, ...) yazıldıkça
``REGISTRY.bind("algoritma_anahtari", fonksiyon)`` ile bağlanacaktır.

Dalga boyları: yedi kategori görünür spektrumu kırmızıdan (680 nm) mora
(410 nm) doğru eşit aralıklarla kaplar; arayüzün renk sistemi buradan gelir.
"""

from __future__ import annotations

from processing import filters, intensity
from processing.registry import (
    Algorithm,
    AlgorithmRegistry,
    BoolParam,
    Category,
    ChoiceParam,
    FloatParam,
    IntParam,
)

_INTERPOLATION = ChoiceParam(
    "interpolation",
    "Interpolation",
    (("nearest", "Nearest"), ("linear", "Linear"), ("cubic", "Cubic")),
    "linear",
)
_KERNEL_SHAPE = ChoiceParam(
    "shape",
    "Structuring element",
    (("rect", "Rect"), ("ellipse", "Ellipse"), ("cross", "Cross")),
    "rect",
)
_KERNEL = IntParam("ksize", "Kernel size", 3, 1, 31, odd_only=True, suffix="px")
_ITERATIONS = IntParam("iterations", "Iterations", 1, 1, 10)

CATEGORIES: tuple[Category, ...] = (
    Category("transform", 1, "Image Transformations", "Geometry · resampling", 680),
    Category("intensity", 2, "Intensity Transformations", "Point operations · tone", 625),
    Category("spatial", 3, "Spatial Filtering", "Convolution · neighbourhood", 580),
    Category("morphology", 4, "Morphological Processing", "Structure · shape", 535),
    Category("segmentation", 5, "Image Segmentation", "Regions · thresholds", 495),
    Category("color", 6, "Color Processing", "Color spaces · channels", 455),
    Category("features", 7, "Feature Extraction", "Edges · corners · keypoints", 415),
)

ALGORITHMS: tuple[Algorithm, ...] = (
    # 01 — Image Transformations
    Algorithm("rotate", "Rotate", "transform", "Rotates the image around its centre.",
              (IntParam("angle", "Angle", 0, -180, 180, suffix="°"), _INTERPOLATION,
               BoolParam("expand", "Expand canvas", True))),
    Algorithm("scale", "Scale", "transform", "Resizes the image by a uniform factor.",
              (FloatParam("factor", "Scale factor", 1.0, 0.1, 4.0, step=0.05, suffix="×"),
               _INTERPOLATION)),
    Algorithm("flip", "Flip", "transform", "Mirrors the image along an axis.",
              (ChoiceParam("axis", "Axis", (("h", "Horizontal"), ("v", "Vertical"), ("both", "Both")), "h"),)),
    Algorithm("translate", "Translate", "transform", "Shifts the image by an offset.",
              (IntParam("dx", "Offset X", 0, -500, 500, suffix="px"),
               IntParam("dy", "Offset Y", 0, -500, 500, suffix="px"))),
    # 02 — Intensity Transformations
    Algorithm("negative", "Negative", "intensity",
              "Inverts intensities: s = (L − 1) − r. Alpha is preserved."),
    Algorithm("log", "Log Transform", "intensity", "Expands dark values: s = c · log(1 + r).",
              (FloatParam("c", "Constant c", 1.0, 0.1, 5.0, step=0.1),)),
    Algorithm("gamma", "Gamma Correction", "intensity",
              "Power law on normalised intensities: s = 255 · c · (r/255)^γ. "
              "γ < 1 brightens shadows, γ > 1 darkens highlights.",
              (FloatParam("gamma", "Gamma γ", 1.0, 0.05, 10.0, step=0.05,
                          help="γ < 1 brightens, γ > 1 darkens; γ = 1 is the identity."),
               FloatParam("c", "Constant c", 1.0, 0.1, 3.0, step=0.05,
                          help="Gain; results above 255 are clipped."))),
    Algorithm("contrast_stretch", "Contrast Stretching", "intensity",
              "Linear min–max stretch: s = 255 · (r − r_min) / (r_max − r_min). "
              "AUTO reads r_min/r_max from the image; MANUAL uses the inputs below.",
              (ChoiceParam("mode", "Range", (("auto", "Auto"), ("manual", "Manual")), "auto"),
               FloatParam("saturate", "Auto · saturate tails", 0.0, 0.0, 5.0, step=0.1, decimals=1, suffix="%",
                          help="AUTO only: ignore this percentage of the darkest and brightest pixels."),
               IntParam("low", "Manual · input low", 0, 0, 255, help="MANUAL only: maps to 0."),
               IntParam("high", "Manual · input high", 255, 0, 255, help="MANUAL only: maps to 255."))),
    Algorithm("hist_eq", "Histogram Equalization", "intensity",
              "CDF-based remapping for a flatter histogram. Color images are equalized on the "
              "luminance channel only, so hues are kept.",
              (ChoiceParam("method", "Method", (("global", "Global"), ("clahe", "CLAHE")), "global"),
               ChoiceParam("space", "Color · luminance channel",
                           (("ycrcb", "YCrCb · Y"), ("lab", "Lab · L*"), ("hsv", "HSV · V")), "ycrcb",
                           help="Ignored for grayscale images."),
               FloatParam("clip", "CLAHE · clip limit", 2.0, 0.5, 10.0, step=0.5, decimals=1),
               IntParam("tiles", "CLAHE · tile grid", 8, 2, 16, help="Tiles per side (N × N)."))),
    # 03 — Spatial Filtering
    Algorithm("mean_blur", "Mean Filter", "spatial",
              "Normalized box kernel: each pixel becomes the average of its k×k neighbourhood "
              "(weights 1/k²). Larger kernels smooth more strongly but also blur edges.",
              (IntParam("ksize", "Kernel size k", 3, 1, 31, odd_only=True, suffix="px",
                        help="Odd k×k window. k = 1 leaves the image unchanged."),)),
    Algorithm("gaussian_blur", "Gaussian Filter", "spatial",
              "Gaussian convolution, w ∝ exp(−(s² + t²) / 2σ²), normalized to sum 1. Smooths with "
              "less ringing than the box filter; choose k ≈ 6σ + 1 to cover ±3σ.",
              (IntParam("ksize", "Kernel size k", 5, 1, 31, odd_only=True, suffix="px",
                        help="Odd k×k window; a kernel smaller than 6σ + 1 is truncated."),
               FloatParam("sigma_x", "Sigma X σ", 1.0, 0.1, 10.0, step=0.1, suffix="px",
                          help="Standard deviation in pixels (σY = σX, isotropic). Must be > 0."))),
    Algorithm("median_blur", "Median Filter", "spatial",
              "Non-linear order-statistic filter: each pixel becomes the median of its k×k "
              "neighbourhood. Removes salt-and-pepper noise while keeping edges sharper than averaging.",
              (IntParam("ksize", "Kernel size k", 3, 3, 31, odd_only=True, suffix="px",
                        help="Odd k ≥ 3 (OpenCV medianBlur requirement)."),)),
    Algorithm("sobel_x", "Sobel X", "spatial",
              "Horizontal derivative Gx ≈ ∂f/∂x — highlights VERTICAL edges. Computed in float on "
              "luminance Y = 0.299R + 0.587G + 0.114B; signs are kept until display.",
              (IntParam("ksize", "Aperture", 3, 1, 7, odd_only=True,
                        help="Sobel kernel size: 1, 3, 5 or 7."),
               ChoiceParam("display", "Display", (("abs", "Magnitude"), ("signed", "Signed")), "abs",
                           help="MAGNITUDE: |Gx| scaled so the strongest edge is 255.  "
                                "SIGNED: 0 → mid-gray, positive brighter, negative darker."))),
    Algorithm("sobel_y", "Sobel Y", "spatial",
              "Vertical derivative Gy ≈ ∂f/∂y — highlights HORIZONTAL edges. Computed in float on "
              "luminance Y = 0.299R + 0.587G + 0.114B; signs are kept until display.",
              (IntParam("ksize", "Aperture", 3, 1, 7, odd_only=True,
                        help="Sobel kernel size: 1, 3, 5 or 7."),
               ChoiceParam("display", "Display", (("abs", "Magnitude"), ("signed", "Signed")), "abs",
                           help="MAGNITUDE: |Gy| scaled so the strongest edge is 255.  "
                                "SIGNED: 0 → mid-gray, positive brighter, negative darker."))),
    Algorithm("laplacian", "Laplacian Filter", "spatial",
              "Second derivative ∇²f = ∂²f/∂x² + ∂²f/∂y² on luminance. Isotropic: responds to "
              "intensity changes in every direction; edges lie at its zero-crossings.",
              (ChoiceParam("neighbours", "Kernel", (("4", "4-neighbour"), ("8", "8-neighbour")), "4",
                           help="4: [0 1 0; 1 −4 1; 0 1 0]   8: [1 1 1; 1 −8 1; 1 1 1]"),
               ChoiceParam("display", "Display", (("abs", "Magnitude"), ("signed", "Signed")), "abs",
                           help="MAGNITUDE: |∇²f| scaled so the strongest response is 255.  "
                                "SIGNED: 0 → mid-gray, so zero-crossings are visible."))),
    Algorithm("unsharp", "Unsharp Masking", "spatial", "Adds back the difference from a blurred copy.",
              (FloatParam("amount", "Amount", 1.0, 0.0, 5.0), FloatParam("sigma", "Sigma σ", 1.5, 0.1, 10.0))),
    # 04 — Morphological Processing
    Algorithm("erode", "Erosion", "morphology", "Shrinks bright regions.", (_KERNEL_SHAPE, _KERNEL, _ITERATIONS)),
    Algorithm("dilate", "Dilation", "morphology", "Grows bright regions.", (_KERNEL_SHAPE, _KERNEL, _ITERATIONS)),
    Algorithm("opening", "Opening", "morphology", "Erosion followed by dilation; removes specks.",
              (_KERNEL_SHAPE, _KERNEL)),
    Algorithm("closing", "Closing", "morphology", "Dilation followed by erosion; fills gaps.",
              (_KERNEL_SHAPE, _KERNEL)),
    Algorithm("gradient", "Morphological Gradient", "morphology", "Dilation minus erosion: object outlines.",
              (_KERNEL_SHAPE, _KERNEL)),
    # 05 — Image Segmentation
    Algorithm("threshold", "Global Threshold", "segmentation", "Binary split at a fixed intensity.",
              (IntParam("thresh", "Threshold", 127, 0, 255), BoolParam("invert", "Invert output"))),
    Algorithm("otsu", "Otsu Threshold", "segmentation", "Chooses the threshold that best separates classes.",
              (BoolParam("invert", "Invert output"),)),
    Algorithm("adaptive", "Adaptive Threshold", "segmentation", "Local thresholds for uneven lighting.",
              (ChoiceParam("method", "Method", (("mean", "Mean"), ("gaussian", "Gaussian")), "gaussian"),
               IntParam("block", "Block size", 11, 3, 99, odd_only=True, suffix="px"),
               IntParam("c", "Constant C", 2, -20, 20))),
    Algorithm("kmeans", "K-Means Clustering", "segmentation", "Groups pixels into k color clusters.",
              (IntParam("k", "Clusters k", 4, 2, 16), IntParam("attempts", "Attempts", 3, 1, 10))),
    # 06 — Color Processing
    Algorithm("color_space", "Color Space Conversion", "color", "Converts between color models.",
              (ChoiceParam("space", "Target space",
                           (("gray", "Gray"), ("hsv", "HSV"), ("lab", "Lab"), ("ycrcb", "YCrCb")), "gray"),)),
    Algorithm("channel", "Channel Isolation", "color", "Shows a single color channel.",
              (ChoiceParam("channel", "Channel", (("r", "Red"), ("g", "Green"), ("b", "Blue")), "r"),
               BoolParam("as_gray", "Display as grayscale", True))),
    Algorithm("hsv_adjust", "HSV Adjust", "color", "Shifts hue and scales saturation / value.",
              (IntParam("hue", "Hue shift", 0, -90, 90, suffix="°"),
               FloatParam("saturation", "Saturation", 1.0, 0.0, 3.0, step=0.05),
               FloatParam("value", "Value", 1.0, 0.0, 3.0, step=0.05))),
    Algorithm("pseudocolor", "Pseudocolor", "color", "Maps intensities onto a color map.",
              (ChoiceParam("cmap", "Color map",
                           (("jet", "Jet"), ("viridis", "Viridis"), ("inferno", "Inferno"),
                            ("ocean", "Ocean"), ("bone", "Bone")), "viridis"),)),
    # 07 — Feature Extraction
    Algorithm("sobel", "Sobel Edges", "features", "First-derivative gradient magnitude.",
              (ChoiceParam("direction", "Direction", (("x", "X"), ("y", "Y"), ("xy", "Magnitude")), "xy"),
               IntParam("ksize", "Aperture", 3, 1, 7, odd_only=True))),
    Algorithm("canny", "Canny Edges", "features", "Multi-stage edge detector with hysteresis.",
              (IntParam("low", "Low threshold", 50, 0, 500), IntParam("high", "High threshold", 150, 0, 500),
               BoolParam("l2", "L2 gradient"))),
    Algorithm("harris", "Harris Corners", "features", "Detects corners from the structure tensor.",
              (IntParam("block", "Block size", 2, 1, 10), FloatParam("k", "Harris k", 0.04, 0.01, 0.2, step=0.01))),
    Algorithm("contours", "Contours", "features", "Traces object boundaries on a binary image.",
              (ChoiceParam("mode", "Retrieval", (("external", "External"), ("tree", "All")), "external"),
               IntParam("thickness", "Line width", 2, 1, 10, suffix="px"))),
    Algorithm("orb", "ORB Keypoints", "features", "Fast oriented keypoints with binary descriptors.",
              (IntParam("n", "Max features", 500, 10, 5000, step=10),)),
)


# Uygulanmış algoritmalar: anahtar → processing modülündeki fonksiyon.
# Yeni bir algoritma yazıldığında yalnızca buraya bir satır eklenir.
IMPLEMENTATIONS = {
    "negative": intensity.negative,
    "gamma": intensity.gamma_correction,
    "contrast_stretch": intensity.contrast_stretch,
    "hist_eq": intensity.equalize_histogram,
    "mean_blur": filters.mean_filter,
    "gaussian_blur": filters.gaussian_filter,
    "median_blur": filters.median_filter,
    "sobel_x": filters.sobel_x,
    "sobel_y": filters.sobel_y,
    "laplacian": filters.laplacian,
}


def build_registry() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    for category in CATEGORIES:
        registry.add_category(category)
    for algorithm in ALGORITHMS:
        registry.add(algorithm)
    for key, func in IMPLEMENTATIONS.items():
        registry.bind(key, func)
    return registry


REGISTRY = build_registry()
