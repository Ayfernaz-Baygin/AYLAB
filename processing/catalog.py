"""Ders kapsamındaki kategori ve algoritma tanımları.

Burada yalnızca *tanımlar* (ad, açıklama, parametreler) vardır. İşlevler
ilgili modüllerde (filters.py, morphology.py, ...) yazıldıkça
``REGISTRY.bind("algoritma_anahtari", fonksiyon)`` ile bağlanacaktır.

Dalga boyları: yedi kategori görünür spektrumu kırmızıdan (680 nm) mora
(410 nm) doğru eşit aralıklarla kaplar; arayüzün renk sistemi buradan gelir.
"""

from __future__ import annotations

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
    Algorithm("negative", "Negative", "intensity", "Inverts intensities: s = L − 1 − r."),
    Algorithm("log", "Log Transform", "intensity", "Expands dark values: s = c · log(1 + r).",
              (FloatParam("c", "Constant c", 1.0, 0.1, 5.0, step=0.1),)),
    Algorithm("gamma", "Gamma Correction", "intensity", "Power-law transform: s = c · r^γ.",
              (FloatParam("gamma", "Gamma γ", 1.0, 0.1, 5.0, step=0.05),
               FloatParam("c", "Constant c", 1.0, 0.1, 3.0, step=0.1))),
    Algorithm("contrast_stretch", "Contrast Stretching", "intensity",
              "Maps an input range linearly to the full output range.",
              (IntParam("low", "Input low", 0, 0, 255), IntParam("high", "Input high", 255, 0, 255))),
    Algorithm("hist_eq", "Histogram Equalization", "intensity",
              "Redistributes intensities for a flatter histogram.",
              (ChoiceParam("method", "Method", (("global", "Global"), ("clahe", "CLAHE")), "global"),
               FloatParam("clip", "CLAHE clip limit", 2.0, 0.5, 10.0, step=0.5))),
    # 03 — Spatial Filtering
    Algorithm("mean_blur", "Mean Filter", "spatial", "Box filter: averages each neighbourhood.", (_KERNEL,)),
    Algorithm("gaussian_blur", "Gaussian Filter", "spatial", "Weighted smoothing with a Gaussian kernel.",
              (IntParam("ksize", "Kernel size", 5, 1, 31, odd_only=True, suffix="px"),
               FloatParam("sigma", "Sigma σ", 1.0, 0.1, 10.0, step=0.1))),
    Algorithm("median_blur", "Median Filter", "spatial", "Non-linear filter; removes salt-and-pepper noise.",
              (_KERNEL,)),
    Algorithm("laplacian", "Laplacian Sharpen", "spatial", "Second-derivative based sharpening.",
              (IntParam("ksize", "Aperture", 3, 1, 7, odd_only=True), FloatParam("weight", "Strength", 1.0, 0.0, 3.0))),
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


def build_registry() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    for category in CATEGORIES:
        registry.add_category(category)
    for algorithm in ALGORITHMS:
        registry.add(algorithm)
    return registry


REGISTRY = build_registry()
