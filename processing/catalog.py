"""Ders kapsamındaki kategori ve algoritma tanımları.

Burada yalnızca *tanımlar* (ad, açıklama, parametreler) vardır. İşlevler
ilgili modüllerde (filters.py, morphology.py, ...) yazıldıkça
``REGISTRY.bind("algoritma_anahtari", fonksiyon)`` ile bağlanacaktır.

Dalga boyları: yedi kategori görünür spektrumu kırmızıdan (680 nm) mora
(410 nm) doğru eşit aralıklarla kaplar; arayüzün renk sistemi buradan gelir.
"""

from __future__ import annotations

from processing import color_processing, filters, intensity, morphology, segmentation
from processing.registry import (
    Algorithm,
    AlgorithmRegistry,
    BoolParam,
    Category,
    ChoiceParam,
    FloatParam,
    IntParam,
    MaskParam,
)

_INTERPOLATION = ChoiceParam(
    "interpolation",
    "Interpolation",
    (("nearest", "Nearest"), ("linear", "Linear"), ("cubic", "Cubic")),
    "linear",
)
# 04 — Morfoloji: ortak parametreler (ikili görüntü + yapı elemanı)
_BINARY_INPUT = (
    ChoiceParam("binarize", "Binarization", (("otsu", "Otsu"), ("manual", "Manual")), "otsu",
                help="How the input becomes the binary set A. OTSU chooses T automatically; "
                     "MANUAL uses T below. Color images use luminance Y = 0.299R + 0.587G + 0.114B."),
    IntParam("thresh", "Manual · threshold T", 127, 0, 255, help="MANUAL only."),
    ChoiceParam("foreground", "Foreground", (("bright", "Bright"), ("dark", "Dark")), "bright",
                help="BRIGHT: object = Y > T.  DARK: object = Y ≤ T (dark objects on a light "
                     "background, e.g. cells or text)."),
)


# 05 — Segmentasyon / eşikleme: ortak parametreler (sınırlar laboratuvar kontrolleriyle aynı)
_SEG_FOREGROUND = ChoiceParam(
    "foreground", "Foreground", (("bright", "Bright · f > T"), ("dark", "Dark · f ≤ T")), "bright",
    help="BRIGHT: object (255) where f > T.  DARK: object where f ≤ T, e.g. dark text on paper — the "
         "complement (inverted mask). Color images use luminance Y = 0.299R + 0.587G + 0.114B.")
_SEG_WINDOW = IntParam("window", "Window", 15, *segmentation.BLOCK_RANGE, odd_only=True, suffix="px",
                       help="Side of the square window that gives m and s (odd).")
_SEG_K = FloatParam("k", "k", 0.2, *segmentation.K_RANGE, step=0.01, decimals=2)


# 06 — Renk işleme: modeller slayt sırasıyla (24–39)
_COLOR_MODELS = (("rgb", "RGB"), ("cmy", "CMY"), ("cmyk", "CMYK"), ("hsv", "HSV"), ("hsl", "HSL"), ("hsi", "HSI"),
                 ("lab", "CIELAB"), ("ycbcr", "YCbCr"))
_CHANNEL_SETS = tuple((m, tuple((str(k + 1), f"{k + 1} · {s.name}") for k, s in enumerate(color_processing.CHANNELS[m])))
                      for m in color_processing.MODELS)
_COLORMAP_CHOICES = tuple((m, color_processing.COLORMAP_NAMES[m]) for m in color_processing.COLORMAPS)


# 05 — Segmentasyon / kenar tabanlı: ortak parametreler
def _edge_threshold(default: float = 0.3, lo: float = 0.05, label: str = "Edge threshold · T / max M") -> FloatParam:
    return FloatParam("t", label, default, lo, 1.0, step=0.01, decimals=2,
                      help="EDGES display: pixels with M ≥ T · max M (fraction of the largest magnitude).")


def _gradient_params() -> tuple:
    return (
        ChoiceParam("display", "Display", (("magnitude", "Magnitude M"), ("gx", "gx"), ("gy", "gy"),
                                           ("direction", "Direction α"), ("edges", "Edges M ≥ T")), "magnitude",
                    help="M: scaled min–max.  gx / gy: signed, 128 = 0 (x points right, y down).  "
                         "DIRECTION: hue = α = atan2(gy, gx), brightness = M.  EDGES: binary M ≥ T."),
        ChoiceParam("magnitude", "Magnitude", (("l2", "√(gx² + gy²)"), ("abs", "|gx| + |gy|")), "l2"),
        ChoiceParam("smoothing", "Pre-smoothing", (("none", "None"), ("gauss5", "5×5 Gaussian")), "none",
                    help="5×5 GAUSSIAN (σ = 1.1) before the derivative, as in the lab — reduces noise edges."),
        _edge_threshold(),
    )


def _seg_block_params() -> tuple:
    return (IntParam("block", "Block size b", 15, *segmentation.BLOCK_RANGE, odd_only=True, suffix="px",
                     help="Side of the neighbourhood (odd)."),
            IntParam("c", "C · subtracted from the mean", 7, *segmentation.C_RANGE,
                     help="Larger C lowers T(x, y): more pixels pass f > T."),
            _SEG_FOREGROUND)


def _se_params(size: int = 3) -> tuple:
    """Yapı elemanı parametreleri; tüm morfoloji işlemleri aynı etiket ve sınırları paylaşır."""
    return (
        ChoiceParam("shape", "SE shape", (("rect", "Rectangle"), ("ellipse", "Ellipse"), ("cross", "Cross")), "rect",
                    help="RECTANGLE: all ones (width = height → square).  ELLIPSE: inscribed disk.  "
                         "CROSS: centre row + centre column."),
        IntParam("width", "SE width", size, 1, 31, odd_only=True, suffix="px"),
        IntParam("height", "SE height", size, 1, 31, odd_only=True, suffix="px",
                 help="1 × k or k × 1 gives a line SE (directional morphology)."),
        IntParam("anchor_x", "Anchor X · origin column", -1, -1, 30,
                 help="−1 = centre. Otherwise 0 … width − 1 (left = 0)."),
        IntParam("anchor_y", "Anchor Y · origin row", -1, -1, 30,
                 help="−1 = centre. Otherwise 0 … height − 1 (top = 0)."),
    )


_STRUCTURING_ELEMENT = _se_params(3)
# Top-hat / black-hat: SE, çıkarılacak ayrıntılardan BÜYÜK olmalı (slayt 44: 40 × 40)
_HAT_SE = _se_params(15)
_MORPH = _BINARY_INPUT + _STRUCTURING_ELEMENT
# Gri seviye morfoloji (slayt 82–85): laboratuvarın SE seçenekleri; boyut 3 / 5 (varsayılan 3)
_GRAY_SE = (
    ChoiceParam("shape", "SE shape", (("rect", "Rectangle"), ("cross", "Cross"), ("ellipse", "Ellipse"),
                                      ("hline", "Horizontal line"), ("vline", "Vertical line"), ("custom", "Custom")),
                "rect", help="Flat SE (the lab's options). ELLIPSE 3 × 3 equals the cross (OpenCV disk).  LINES: "
                             "1 × n / n × 1.  CUSTOM: drawn in the grid below, with its own origin."),
    IntParam("size", "SE size n", 3, 3, 31, odd_only=True, suffix="px",
             help="n × n window (the lab uses 3 and 5). Ignored for CUSTOM. The origin is the centre."),
    MaskParam("custom", "Custom SE · grid", "010/011/000@1,1", max_size=7,
              help="CUSTOM only. Left click toggles a cell, right click moves the origin (it may lie outside "
                   "the cells). Dilation uses the reflected SE b̂, so an asymmetric SE shifts the result."),
)
_ITERATIONS = IntParam("iterations", "Iterations", 1, 1, 10, help="Applies the operation n times with the same SE.")

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
    # 04 — Morphological Processing (binary; pixels outside the image are background)
    Algorithm("erode", "Erosion", "morphology",
              "A ⊖ B = { z | (B)z ⊆ A }: a pixel stays foreground only if the whole SE, placed at its "
              "origin, FITS inside the object. Shrinks objects, removes specks, separates touching shapes.",
              _MORPH + (_ITERATIONS,)),
    Algorithm("dilate", "Dilation", "morphology",
              "A ⊕ B = { z | (B̂)z ∩ A ≠ ∅ }: a pixel becomes foreground if the reflected SE HITS the "
              "object. Grows objects, bridges gaps, fills small holes.",
              _MORPH + (_ITERATIONS,)),
    Algorithm("opening", "Opening", "morphology",
              "A ∘ B = (A ⊖ B) ⊕ B — erosion, then dilation. Removes objects and protrusions smaller "
              "than the SE while keeping the shape of larger ones. Idempotent.",
              _MORPH),
    Algorithm("closing", "Closing", "morphology",
              "A • B = (A ⊕ B) ⊖ B — dilation, then erosion. Fills holes and gaps smaller than the SE "
              "without growing the objects overall. Idempotent; A ⊆ A • B.",
              _MORPH),
    Algorithm("gradient", "Morphological Gradient", "morphology",
              "(A ⊕ B) − (A ⊖ B): the band between dilation and erosion — a shape-based, non-linear "
              "edge detector that outlines every object.",
              _MORPH),
    Algorithm("hit_or_miss", "Hit-or-Miss Transform", "morphology",
              "A ⊛ B = (A ⊖ B1) ∩ (Aᶜ ⊖ B2): marks positions where the 3 × 3 pattern matches exactly — "
              "B1 (1) must be foreground, B2 (−1) must be background, 0 = don't care.",
              _BINARY_INPUT + (
                  ChoiceParam("pattern", "Pattern",
                              (("corner", "Upper-right corner · slide 38b"), ("edge", "Right edge · slide 38c"),
                               ("hole", "Isolated hole · slide 38a"), ("cv_example", "OpenCV example · slide 39"),
                               ("isolated", "Isolated point"), ("end_point", "Line end · east")), "corner",
                              help="corner [−1 −1 −1; 1 1 −1; 1 1 −1]   edge [0 0 −1; 1 1 −1; 0 0 −1]   "
                                   "hole [1 1 1; 1 −1 1; 1 1 1]"),
                  ChoiceParam("rotation", "Rotate pattern",
                              (("0", "0°"), ("90", "90°"), ("180", "180°"), ("270", "270°")), "0",
                              help="Counter-clockwise; e.g. the upper-right corner rotated 90° finds "
                                   "upper-left corners."),
                  ChoiceParam("display", "Display", (("matches", "Matches"), ("overlay", "On shape")), "matches",
                              help="MATCHES: matched positions only (255).  ON SHAPE: matches 255 drawn "
                                   "over the binary shape in dark gray (70)."))),
    Algorithm("top_hat", "White Top-Hat", "morphology",
              "T_hat(f) = f − (f ∘ b): grayscale image minus its opening. Extracts small BRIGHT details "
              "that the opening removes (structures smaller than the SE); the slowly varying background "
              "drops to ≈ 0. Grayscale intensities are used directly — no thresholding. Values are shown "
              "unscaled; color images use luminance Y. Image border: pixels outside the image are ignored, so "
              "a small bright detail that touches the border can survive the opening and is then NOT extracted "
              "(top-hat ≈ 0 there).",
              _HAT_SE),
    Algorithm("black_hat", "Black-Hat", "morphology",
              "B_hat(f) = (f • b) − f: closing of the grayscale image minus the image (bottom-hat). Extracts "
              "small DARK details that the closing fills in. No thresholding; values are shown unscaled; "
              "color images use luminance Y.",
              _HAT_SE),
    Algorithm("hat_correction", "Illumination / Contrast Correction", "morphology",
              "corrected = img + top_hat − black_hat (same SE for both). Brightens small bright details and "
              "darkens small dark details, raising local contrast. Computed in int32 and clipped to 0 … 255. "
              "Color: Δ = T_hat(Y) − B_hat(Y) is added to B, G and R equally, so chroma is preserved.",
              _HAT_SE),
    Algorithm("boundary", "Boundary Extraction", "morphology",
              "Extracts object boundaries by subtracting the eroded foreground from the original binary "
              "image: β(A) = A − (A ⊖ B). Only foreground pixels removed by the erosion remain; a larger SE "
              "gives a thicker boundary.",
              _MORPH),
    Algorithm("hole_filling", "Hole Filling", "morphology",
              "Fills holes — background regions completely enclosed by the foreground. FLOOD FILL (slide 59): "
              "flood the background from the image border (4-connected), invert, OR with A — every closed "
              "hole at once. RECONSTRUCTION (slides 56–58): X_k = (X_k−1 ⊕ B) ∩ Aᶜ from a seed until "
              "X_k = X_k−1, result X_k ∪ A — only seeded holes fill. Gaps open to the border are never filled.",
              _BINARY_INPUT + (
                  ChoiceParam("method", "Method", (("flood", "Flood fill"), ("reconstruction", "Reconstruction")),
                              "flood", help="FLOOD FILL: all closed holes at once.  RECONSTRUCTION: iterative "
                                            "conditional dilation from seeds; only the seeded hole(s)."),
                  ChoiceParam("seeds", "Reconstruction · seeds",
                              (("largest", "Largest"), ("centre", "Centre"), ("all", "All")),
                              "largest", help="RECONSTRUCTION only. LARGEST: the largest hole.  CENTRE: the hole "
                                              "nearest the image centre.  ALL: every hole (same result as flood "
                                              "fill). One seed per hole, at the hole pixel nearest its centroid."),
                  ChoiceParam("shape", "Reconstruction · SE 3 × 3", (("cross", "Cross"), ("rect", "Square")),
                              "cross", help="CROSS (textbook): holes are 4-connected background.  SQUARE: "
                                            "8-connected — a hole that touches the outside only diagonally is "
                                            "not closed and is not filled."),
              )),
    Algorithm("connected_components", "Connected Components", "morphology",
              "Labels each separate object of the binary image A with its own integer (background = 0) and "
              "lists its area, bounding box and centroid. Course method: X_k = (X_k−1 ⊕ B) ∩ A grown from an "
              "unlabelled foreground pixel until X_k = X_k−1. B = 3×3 cross → 4-connected; 3×3 square → "
              "8-connected (diagonal neighbours join).",
              _BINARY_INPUT + (
                  ChoiceParam("connectivity", "Connectivity · SE B", (("4", "4 · Cross"), ("8", "8 · Square")), "8",
                              help="4: B = 3×3 cross — only edge neighbours connect.  8: B = 3×3 square — "
                                   "diagonal neighbours connect too."),
                  ChoiceParam("display", "Display", (("labels", "Labels"), ("boxes", "Labels + boxes")), "labels",
                              help="Each label gets its own colour (background black); colours may repeat "
                                   "for many components.  BOXES also draws bounding boxes (1 px outside the "
                                   "object) and centroids for components of at least 3 × 3 px."),
              )),
    Algorithm("convex_hull", "Convex Hull", "morphology",
              "The smallest convex region enclosing each object — a rubber band stretched around it (slide 63). "
              "CONTOURS (slide 65): external contours → convex hull polygon; holes do not change the hull. "
              "HIT-OR-MISS (Fig. 9.21): X_k = (X_k−1 ⊛ Bⁱ) ∪ X_k−1 for B¹…B⁴, C(A) = ∪ Dⁱ, limited to each "
              "object's bounding box.",
              _BINARY_INPUT + (
                  ChoiceParam("method", "Method", (("contour", "Contours"), ("morph", "Hit-or-miss")), "contour",
                              help="CONTOURS: exact convex polygon of each object (OpenCV findContours + "
                                   "convexHull).  HIT-OR-MISS: textbook iteration with B¹…B⁴ — a digital "
                                   "approximation that fills towards the bounding box."),
                  ChoiceParam("display", "Display", (("filled", "Filled"), ("overlay", "Overlay")), "filled",
                              help="FILLED: binary C(A) ∪ A.  OVERLAY: object grey, area added by the hull blue, "
                                   "external contour green, hull boundary yellow."),
              )),
    Algorithm("thinning", "Thinning", "morphology",
              "A ⊗ B = A − (A ⊛ B), applied with B¹…B⁸ in sequence and repeated until no pixel changes "
              "(Fig. 9.23). Peels the boundary until only a thin curve near the object's middle remains. "
              "The result is usually about one pixel wide and usually keeps the objects connected, but this is "
              "not guaranteed: small 2 × 2 blocks can remain (both methods), and very small specks can "
              "disappear completely (Zhang–Suen).",
              _BINARY_INPUT + (
                  ChoiceParam("method", "Method", (("hmt", "Hit-or-miss"), ("zhang_suen", "Zhang–Suen")), "hmt",
                              help="HIT-OR-MISS: textbook B¹…B⁸ sequence.  ZHANG–SUEN: two-sub-step parallel "
                                   "thinning (slide 68)."),
                  IntParam("max_cycles", "Max cycles · 0 = converge", 0, 0, 500,
                           help="0 runs until X no longer changes. A value n stops after n full cycles "
                                "(useful to watch the boundary being peeled)."),
              )),
    Algorithm("thickening", "Thickening", "morphology",
              "Dual of thinning (Fig. 9.24): thin the background Aᶜ with B¹…B⁸, then complement — objects grow "
              "where the SE hits the background pattern, but neighbouring objects never merge (the "
              "background skeleton separates them). The textbook cleanup of disconnected parts is applied but "
              "normally finds nothing here, because outside the image counts as fixed background.",
              _BINARY_INPUT + (
                  IntParam("max_cycles", "Max cycles · 0 = converge", 10, 0, 500,
                           help="Each cycle grows the objects by about one pixel. 0 runs until convergence "
                                "(objects meet at the background skeleton) — slow on large images."),
                  ChoiceParam("cleanup", "Disconnected parts", (("remove", "Remove"), ("keep", "Keep")), "remove",
                              help="REMOVE (textbook post-processing): drop grown parts that do not touch any "
                                   "original object. Outside the image counts as fixed background, so no "
                                   "bands grow from the border and this step normally finds nothing."),
              )),
    Algorithm("skeletonization", "Skeletonization", "morphology",
              "Morphological skeleton (Fig. 9.26): S_k(A) = (A ⊖ kB) − (A ⊖ kB) ∘ B for k = 0…K, "
              "K = max{k | A ⊖ kB ≠ ∅}, S(A) = ∪ S_k(A). With the k values the shape is rebuilt exactly: "
              "A = ∪ (S_k(A) ⊕ kB). The skeleton may be disconnected and need not be one pixel wide.",
              _BINARY_INPUT + _STRUCTURING_ELEMENT[:3] + (
                  ChoiceParam("display", "Display", (("skeleton", "Skeleton S(A)"), ("levels", "Levels · colour = k"),
                                                     ("layer", "One level k"), ("reconstruction", "Reconstruction")),
                              "skeleton",
                              help="SKELETON: S(A) in white.  LEVELS: A dark grey, each S_k in its own colour.  "
                                   "ONE LEVEL: A dark grey, (A ⊖ kB) ∘ B light grey, S_k yellow.  "
                                   "RECONSTRUCTION: ∪ (S_k ⊕ kB), equal to A."),
                  IntParam("level", "One level · k", 0, 0, 200,
                           help="ONE LEVEL only. For k > K the eroded set is empty; the status bar lists "
                                "K and |S_k| for every level. The SE origin is always its centre."),
              )),
    Algorithm("pruning", "Pruning", "morphology",
              "Removes short spurs from a thin set (Fig. 9.27): X1 = A ⊗ {B} (n cycles with the end-point "
              "kernels B¹…B⁸), X2 = ∪ X1 ⊛ Bᵏ (end points), X3 = (X2 ⊕ H) ∩ A n times (H = 3×3), "
              "X4 = X1 ∪ X3. Branches of length ≤ n are removed — including short real branches.",
              _BINARY_INPUT + (
                  IntParam("n", "Spur length n", 3, 1, 50, suffix="px",
                           help="Thinning cycles and regrowth steps. Spurs of up to n pixels vanish; a REAL "
                                "branch of up to n pixels is removed as well, and a spur next to a real "
                                "end point can partly grow back. Apply to a thin input (e.g. after Thinning)."),
                  ChoiceParam("display", "Display", (("result", "Result X4"), ("overlay", "Overlay"), ("x1", "X1 · thinned"),
                                                     ("x2", "X2 · end points"), ("x3", "X3 · regrown")), "result",
                              help="OVERLAY: removed red, X1 white, tips restored by X3 green, end points X2 "
                                   "yellow.  X1/X2/X3: intermediate sets in white."),
              )),
    Algorithm("gray_erosion", "Grayscale Erosion", "morphology",
              "[f ⊖ b](x, y) = min f(x + s, y + t), (s, t) ∈ b — the image is a height surface and the flat SE "
              "takes the MINIMUM under it. Bright details smaller than the SE shrink or vanish, dark ones grow; "
              "the image gets darker. No thresholding (colour: luminance Y).",
              _GRAY_SE),
    Algorithm("gray_dilation", "Grayscale Dilation", "morphology",
              "[f ⊕ b](x, y) = max f(x − s, y − t), (s, t) ∈ b — the MAXIMUM under the REFLECTED SE b̂. Dark "
              "details smaller than the SE shrink or vanish, bright ones grow; the image gets brighter.",
              _GRAY_SE),
    Algorithm("gray_opening", "Grayscale Opening", "morphology",
              "f ∘ b = (f ⊖ b) ⊕ b — erosion, then dilation with the same SE. Removes bright details smaller "
              "than the SE (salt noise, thin bright lines) while keeping the overall brightness; f ∘ b ≤ f. "
              "Image border: pixels outside the image are ignored, so the SE window is cut off there. A small "
              "bright detail that touches the border can therefore survive the opening.",
              _GRAY_SE + (
                  ChoiceParam("display", "Display", (("opening", "f ∘ b"), ("erosion", "Stage 1 · f ⊖ b")), "opening",
                              help="f ∘ b: the result.  STAGE 1: the erosion that the dilation then works on."),
              )),
    Algorithm("gray_closing", "Grayscale Closing", "morphology",
              "f • b = (f ⊕ b) ⊖ b — dilation, then erosion with the same SE. Fills dark details smaller than "
              "the SE (pepper noise, pits) while keeping the overall brightness; f • b ≥ f.",
              _GRAY_SE + (
                  ChoiceParam("display", "Display", (("closing", "f • b"), ("dilation", "Stage 1 · f ⊕ b")), "closing",
                              help="f • b: the result.  STAGE 1: the dilation that the erosion then works on."),
              )),
    Algorithm("gray_gradient", "Grayscale Morphological Gradient", "morphology",
              "g = (f ⊕ b) − (f ⊖ b) — local max minus local min inside the SE window: large exactly where the "
              "intensity changes, an edge map (Fig. 9.41). Never negative when the origin is in b.",
              _GRAY_SE + (
                  ChoiceParam("display", "Display", (("gradient", "Gradient g"), ("dilation", "f ⊕ b"), ("erosion", "f ⊖ b")),
                              "gradient", help="Inspect the two terms separately: g = (f ⊕ b) − (f ⊖ b). The status "
                                               "bar lists the mean of both."),
              )),
    # 05 — Image Segmentation · Thresholding (slides 7–17, Chapter 5 lab)
    Algorithm("threshold", "Manual / Fixed Threshold", "segmentation",
              "g(x, y) = 1 if f(x, y) > T, 0 otherwise — one threshold for the whole image. Works when object "
              "and background occupy different intensity ranges; noise widens the histogram modes until no "
              "single T separates them (slide 9).",
              (IntParam("thresh", "Threshold T", 127, 0, 255),
               _SEG_FOREGROUND)),
    Algorithm("iterative_threshold", "Automatic Global Threshold", "segmentation",
              "Iterative threshold selection (slide 11): choose T₀, split the image, take the means μ₁ (f > T) "
              "and μ₂ (f ≤ T), set T = (μ₁ + μ₂) / 2 and repeat until T changes by less than ΔT (at most "
              "100 iterations). Converges quickly when the histogram has two clear modes.",
              (ChoiceParam("init", "Initial T₀", (("mean", "Image mean"), ("custom", "Chosen value")), "mean",
                           help="IMAGE MEAN: T₀ = average intensity.  CHOSEN VALUE: T₀ from the slider below."),
               IntParam("t0", "T₀ · chosen value", 127, segmentation.T0_RANGE[0], segmentation.T0_RANGE[1],
                        help="CHOSEN VALUE only."),
               FloatParam("delta", "Stop when |ΔT| <", 0.5, *segmentation.DELTA_T_RANGE, step=0.1, decimals=1),
               _SEG_FOREGROUND)),
    Algorithm("otsu", "Otsu's Method", "segmentation",
              "T* = argmax σ²_B(k), σ²_B(k) = (m_G·ω₀ − m)² / (ω₀(1 − ω₀)): tries every threshold k and keeps "
              "the one that makes the two classes most separable (slides 12–13). Needs only the histogram; "
              "robust for bimodal histograms. Heavy noise merges the modes — smoothing first helps. A uniform "
              "image has σ²_B ≡ 0 and gives T* = 0.",
              (_SEG_FOREGROUND,)),
    Algorithm("adaptive_mean", "Adaptive Mean Threshold", "segmentation",
              "T(x, y) = mean of the b × b block around (x, y) − C: every pixel gets its own threshold, so "
              "the decision follows uneven lighting (slides 14–15). Image border: reflected without "
              "repeating the edge pixel (reflect-101).",
              _seg_block_params()),
    Algorithm("adaptive_gaussian", "Adaptive Gaussian Threshold", "segmentation",
              "T(x, y) = Gaussian-weighted mean of the b × b block − C, σ = 0.3·((b − 1)/2 − 1) + 0.8 (as in "
              "OpenCV). Nearer pixels count more than in the plain mean (slides 14–15). Border: reflect-101. "
              "Small differences to cv2.adaptiveThreshold are possible: OpenCV rounds the mean to 8 bits and "
              "repeats the edge pixel.",
              _seg_block_params()),
    Algorithm("moving_average", "Moving Averages Threshold", "segmentation",
              "g = 1 if f > c · m, m = mean of the last n pixels along a zigzag scan (even rows left → right, "
              "odd rows right → left, continuing across rows; slide 16, Gonzalez n = 20, c = 0.5). Very cheap; "
              "suited to text under spotlight or shading. The first n − 1 scanned pixels use the mean of the "
              "pixels seen so far.",
              (IntParam("n", "n · pixels averaged", 20, *segmentation.MOVING_N_RANGE),
               FloatParam("c", "c · fraction of the mean", 0.5, *segmentation.MOVING_C_RANGE, step=0.01, decimals=2),
               _SEG_FOREGROUND)),
    Algorithm("niblack", "Niblack Threshold", "segmentation",
              "T(x, y) = m − k·s, m and s: mean and standard deviation of the window (slide 17, scikit-image "
              "convention). Follows every local variation, so flat noisy background turns into specks. "
              "Border: reflect-101.",
              (_SEG_WINDOW, _SEG_K, _SEG_FOREGROUND)),
    Algorithm("sauvola", "Sauvola Threshold", "segmentation",
              "T(x, y) = m · (1 + k · (s / R − 1)), R: dynamic range of s (slide 17). Where s is small (flat "
              "background) T drops below m, so the background stays clean. Border: reflect-101.",
              (_SEG_WINDOW, _SEG_K,
               IntParam("r", "R · dynamic range of s", 128, *segmentation.R_RANGE),
               _SEG_FOREGROUND)),
    # 05 — Image Segmentation · Edge-based (slides 18–28, Chapter 5 lab)
    Algorithm("edge_discontinuity", "Discontinuities & Derivatives", "segmentation",
              "Edge-based segmentation looks for abrupt intensity changes (slides 18–19). First derivative "
              "f′(x) = f(x+1) − f(x): non-zero along ramps, a single response at a step. Second derivative "
              "f″(x) = f(x+1) + f(x−1) − 2f(x): a double (+/−) response at edges, zero on ramps — strongest on "
              "isolated points and thin lines. Undefined end values are left at 0.",
              (ChoiceParam("order", "Derivative", (("first", "First · f′"), ("second", "Second · f″")), "first"),
               ChoiceParam("axis", "Direction", (("horizontal", "Along rows (x)"), ("vertical", "Along columns (y)")),
                           "horizontal"),
               ChoiceParam("display", "Display", (("signed", "Signed · 128 = 0"), ("abs", "Absolute")), "signed"))),
    Algorithm("edge_point", "Point Detection", "segmentation",
              "R = Σ w·f with the 8-neighbour Laplacian kernel [1 1 1; 1 −8 1; 1 1 1]; g = 1 if |R| ≥ T "
              "(slide 20). An isolated point differs from all eight neighbours, so |R| is largest there. "
              "T is a fraction of max |R| (Gonzalez Fig. 10.4: 90 %). A flat image has R ≡ 0 and gives no points.",
              (FloatParam("t", "T / max |R|", 0.9, 0.1, 1.0, step=0.01, decimals=2),
               ChoiceParam("display", "Display", (("detected", "Detected |R| ≥ T"), ("abs", "|R|"),
                                                  ("signed", "R · signed")), "detected"))),
    Algorithm("edge_line", "Line Detection", "segmentation",
              "R = Σ w·f with a direction-tuned kernel (slide 21): a row of 2s along the direction, −1 elsewhere. "
              "Negative responses (the dark sides of a bright line) are set to 0, then g = 1 if R ≥ T. "
              "+45° is the ╲ diagonal on screen (Gonzalez: x points down).",
              (ChoiceParam("direction", "Kernel", (("horizontal", "Horizontal"), ("p45", "+45°"),
                                                   ("vertical", "Vertical"), ("m45", "−45°")), "p45"),
               FloatParam("t", "T / max R", 0.6, 0.1, 1.0, step=0.01, decimals=2),
               ChoiceParam("display", "Display", (("detected", "R ≥ T"), ("positive", "max(R, 0)"),
                                                  ("signed", "R signed")), "detected",
                           help="R ≥ T: detected line pixels.  MAX(R, 0): response with negatives set to 0.  "
                                "R SIGNED: 128 = 0."))),
    Algorithm("edge_gradient", "Gradient (First-Order Derivative)", "segmentation",
              "∇f = (gx, gy), gx = f(x+1, y) − f(x, y), gy = f(x, y+1) − f(x, y) — the forward difference f′ of "
              "slide 19 along both axes. M(x, y) = √(gx² + gy²) is the edge strength, α = atan2(gy, gx) the "
              "direction of fastest change; the edge runs perpendicular to α (slide 22). Border: reflect-101.",
              _gradient_params()),
    Algorithm("edge_roberts", "Roberts Operator", "segmentation",
              "2 × 2 cross-differences (slide 23): gx = f(x+1, y+1) − f(x, y), gy = f(x, y+1) − f(x+1, y). "
              "Smallest support, most sensitive to noise. M and α as for the gradient.",
              _gradient_params()),
    Algorithm("edge_sobel", "Sobel Operator", "segmentation",
              "3 × 3 kernels with the centre row/column weighted 2 (slide 23): wx = [−1 0 1; −2 0 2; −1 0 1], "
              "wy = its transpose. The weighting also smooths noise. Slides label the [−1 −2 −1; 0 0 0; 1 2 1] "
              "kernel gx (x down); here it is wy (x right), as in the lab.",
              _gradient_params()),
    Algorithm("edge_prewitt", "Prewitt Operator", "segmentation",
              "3 × 3 kernels with equal weights (slide 23): wx = [−1 0 1; −1 0 1; −1 0 1], wy = its transpose. "
              "Simpler than Sobel, slightly less noise suppression.",
              _gradient_params()),
    Algorithm("edge_laplacian", "Laplacian", "segmentation",
              "∇²f with the 4- or 8-neighbour kernel (slides 23–24). Second derivative: a double response at "
              "edges, zero on flat areas and linear ramps; very sensitive to noise. Edges are the zero "
              "crossings: opposite neighbours (horizontal, vertical, diagonals) with u·v < 0 and |u − v| > T. "
              "An ideal one-pixel step gives no crossing under this rule (its + and − responses are adjacent); "
              "smoothing first (LoG) spreads them.",
              (ChoiceParam("kernel", "Laplacian kernel", (("4", "4-neighbour"), ("8", "8-neighbour")), "8"),
               ChoiceParam("display", "Display", (("signed", "∇² · signed"), ("abs", "|∇²|"),
                                                  ("zero_crossings", "Zero crossings")), "signed"),
               FloatParam("t", "Zero-crossing T · / max |∇²|", 0.1, 0.0, 0.5, step=0.01, decimals=2,
                          help="A crossing counts only if |u − v| > T · max |∇²|."))),
    Algorithm("edge_log", "Laplacian of Gaussian (LoG)", "segmentation",
              "Marr–Hildreth: ∇²(G_σ ⋆ f) — smooth with a Gaussian (2⌈3σ⌉ + 1 taps), then apply the Laplacian; "
              "edges are the zero crossings, thin closed contours (slides 23–24). Larger σ suppresses noise and "
              "fine detail. The threshold on |u − v| removes weak crossings.",
              (FloatParam("sigma", "σ", 2.0, 0.5, 5.0, step=0.1, decimals=1),
               ChoiceParam("kernel", "Laplacian kernel", (("4", "4-neighbour"), ("8", "8-neighbour")), "8"),
               ChoiceParam("display", "Display", (("zero_crossings", "Zero crossings"), ("signed", "LoG · signed")),
                           "zero_crossings"),
               FloatParam("t", "Zero-crossing T · / max |∇²|", 0.1, 0.0, 0.5, step=0.01, decimals=2,
                          help="A crossing counts only if |u − v| > T · max |∇²|."))),
    Algorithm("edge_canny", "Canny Edge Detector", "segmentation",
              "Slides 25–26 / lab: (1) Gaussian smoothing σ, (2) Sobel magnitude M (scaled to 0–255) and "
              "direction, (3) non-maximum suppression across the edge (4 sectors), (4) double threshold: "
              "M ≥ T₂ strong, T₁ ≤ M < T₂ weak, (5) hysteresis / edge linking: weak pixels are kept only if "
              "8-connected to a strong one. T₁ must be smaller than T₂. Not identical to cv2.Canny (which "
              "thresholds the unscaled magnitude). NMS keeps ties (M ≥ both neighbours), so a symmetric step can "
              "give a 2-pixel-wide edge.",
              (FloatParam("sigma", "Gaussian σ · 0 = none", 1.4, 0.0, 4.0, step=0.1, decimals=1),
               IntParam("low", "T₁ · low threshold", 10, 1, 150),
               IntParam("high", "T₂ · high threshold", 45, 2, 255),
               ChoiceParam("display", "Display", (("edges", "Edges"), ("nms", "After NMS"),
                                                  ("double", "Strong / weak"), ("magnitude", "Magnitude M"),
                                                  ("smoothed", "Smoothed input")), "edges",
                           help="STRONG / WEAK: strong white, weak orange (before hysteresis)."))),
    Algorithm("edge_linking", "Edge Linking", "segmentation",
              "'Thresholding of gradient direction' (slide 27), implemented as the local-processing method of "
              "Gonzalez & Woods (4th ed., §10.2 — textbook, not detailed in the slides): Sobel M and α; "
              "g = 1 where M ≥ T_M and |α − A| ≤ T_A; along the edge direction θ, gaps of at most K pixels "
              "between edge pixels are filled (the book rotates g by θ and scans rows; here the scan follows "
              "digital lines parallel to θ, without resampling, never leaving the image). θ = 0° horizontal, "
              "90° vertical, counter-clockwise on screen; A = 90° − θ. BOTH: θ = 0° and 90° combined. Angles are "
              "compared modulo 180° (edge polarity ignored, 179° ≈ 0°).",
              (FloatParam("tm", "T_M · / max M", 0.3, 0.05, 1.0, step=0.01, decimals=2),
               IntParam("ta", "T_A · angle tolerance", 45, 0, 90, suffix="°"),
               IntParam("k", "K · longest gap filled", 25, 1, 100, suffix="px"),
               ChoiceParam("orientation", "Edges", (("both", "Both"), ("horizontal", "Horizontal"),
                                                    ("vertical", "Vertical"), ("angle", "Angle θ")), "both",
                           help="BOTH: horizontal + vertical.  θ: link along the edge direction set below."),
               IntParam("angle", "θ · edge direction", 45, 0, 179, suffix="°",
                        help="Used when Edges = θ. 0° horizontal, 90° vertical, 45° rising to the right (╱), 135° (╲)."),
               ChoiceParam("display", "Display", (("linked", "Linked"), ("overlay", "Overlay"),
                                                  ("candidates", "Candidates g")), "linked",
                           help="OVERLAY: candidate edge pixels white, filled gaps orange."))),
    Algorithm("edge_boundary", "Boundary Extraction · Contour Tracing", "segmentation",
              "Closed boundaries of the binary objects (slide 27: 'detect closed contours, form boundary "
              "representations'), traced with the Moore boundary-following algorithm (Gonzalez & Woods §11.1 — "
              "textbook): start at the top-left object pixel, search the 8 neighbours clockwise. Each "
              "8-connected object gives one closed outer boundary; the table lists the objects. (Morphological "
              "β(A) = A − (A ⊖ B) is in Morphological Processing.)",
              _BINARY_INPUT + (
                  ChoiceParam("display", "Display", (("overlay", "Overlay"), ("boundaries", "Boundaries only")),
                              "overlay", help="OVERLAY: image dimmed, each boundary in its object's colour."),
              )),
    Algorithm("edge_hough", "Hough Transform", "segmentation",
              "Global edge linking (slides 27–28): every Canny edge point (σ = 1.4) votes for each line "
              "ρ = x cos θ + y sin θ through it, θ = −90° … 89° in 1° steps, ρ in 1-pixel bins. Peaks of the "
              "accumulator A(ρ, θ) with at least the peak threshold × max votes, local maxima in a ±5 "
              "neighbourhood, are the detected lines.",
              (IntParam("low", "Canny T₁", 20, 1, 150),
               IntParam("high", "Canny T₂", 50, 2, 255),
               FloatParam("frac", "Peaks · / max votes", 0.5, 0.1, 1.0, step=0.01, decimals=2,
                          help="A peak needs at least this fraction of the largest vote count (and ≥ 2 votes)."),
               IntParam("max_lines", "Max lines", 24, 1, 60),
               ChoiceParam("display", "Display", (("lines", "Lines"), ("accumulator", "A(ρ, θ)"),
                                                  ("edges", "Edge points")), "lines",
                           help="LINES: detected lines on the input.  A(ρ, θ): the accumulator — rows = ρ (−D … D), columns = θ (−90° … 89°); size differs "
                                "from the image."))),
    # 05 — Segmentasyon / bölge tabanlı (slayt 29–38; laboratuvar growAll, quadtree, watershed, viewMWS)
    Algorithm("region_growing", "Region Growing", "segmentation",
              "Slides 30–31 / lab: start from seed points and add neighbours that satisfy an intensity-similarity "
              "criterion, layer by layer: a neighbour q of a region pixel joins if |f(q) − f(seed)| ≤ T, where "
              "f(seed) is the value of the seed the region grew from. Stops when no new pixel passes (the region "
              "has stabilized). POINT: one seed — click the image (A or B) to place it, or type x, y. BRIGHTEST "
              "(Gonzalez, weld example): the brightest q % of the pixels are seeds; each 8-connected seed group "
              "grows its own region.",
              (ChoiceParam("seeds", "Seeds", (("point", "Point · click / x, y"), ("brightest", "Brightest q %")),
                           "point", help="POINT: single seed at (x, y).  BRIGHTEST: pixels with f ≥ the top-q % "
                                         "value are seeds (several regions)."),
               IntParam("seed_x", "Seed x · −1 = centre", -1, -1, segmentation.SEED_COORD_MAX, suffix="px",
                        help="POINT only. Click the image to set x and y; −1 = image centre."),
               IntParam("seed_y", "Seed y · −1 = centre", -1, -1, segmentation.SEED_COORD_MAX, suffix="px",
                        help="POINT only. Must lie inside the image."),
               FloatParam("q", "Brightest · top q %", 0.3, *segmentation.GROW_Q_RANGE, step=0.05, decimals=2,
                          suffix="%", help="BRIGHTEST only: seed threshold = the value exceeded by q % of pixels."),
               IntParam("t", "T · |f(q) − f(seed)| ≤ T", 40, *segmentation.GROW_T_RANGE,
                        help="Neighbour acceptance threshold (intensity similarity)."),
               ChoiceParam("connectivity", "Neighbours", (("8", "8-connected"), ("4", "4-connected")), "8"),
               ChoiceParam("display", "Display", (("overlay", "Overlay"), ("binary", "Binary"), ("labels", "Labels"),
                                                  ("difference", "|f − f(seed)|")), "overlay",
                           help="|f − f(seed)| (POINT only): the distance the criterion compares with T."))),
    Algorithm("region_split_merge", "Region Splitting & Merging", "segmentation",
              "Slides 32–34 / lab: quadtree — the root is the whole image; a block that fails the predicate Q is "
              "split into four quadrants (odd sizes give halves that differ by one pixel), down to the minimum "
              "block; adjacent leaves that are similar are then merged, so regions need not be squares. HOMOG: "
              "Q = σ ≤ a, merge when the leaf means differ by ≤ the merge threshold. GONZALEZ (Fig. 10.48): "
              "Q = σ > a and 0 < m < b; as in DIPUM splitmerge, Q = TRUE blocks keep splitting and the Q = TRUE "
              "leaves form the region (white). Max depth is an AYLAB safety limit (not in the lab).",
              (ChoiceParam("predicate", "Predicate Q", (("homog", "Homogeneous"), ("gonzalez", "Gonzalez")), "homog",
                           help="HOMOGENEOUS: Q = σ ≤ a.  GONZALEZ (Fig. 10.48): Q = σ > a and 0 < m < b."),
               IntParam("a", "a · σ limit", 8, 1, 40),
               IntParam("b", "b · mean limit", 125, 20, 255, help="GONZALEZ only."),
               ChoiceParam("min_size", "Min block", tuple((v, f"{v}×{v}") for v in segmentation.SPLIT_MIN_BLOCKS),
                           "4", help="A block is split only while both sides are ≥ 2 × this size."),
               IntParam("max_depth", "Max depth", segmentation.SPLIT_MAX_DEPTH, 0, segmentation.SPLIT_MAX_DEPTH,
                        help="Quadtree levels below the root (0 = no split). 12 never limits images up to 8192 px."),
               IntParam("merge_t", "Merge if |Δmean| ≤", 12, 0, 60, help="HOMOG only."),
               ChoiceParam("display", "Display", (("merged", "Merged"), ("split", "Split · quadtree"),
                                                  ("labels", "Labels")), "merged",
                           help="MERGED: regions over the leaf means, white borders (GONZALEZ: binary).  SPLIT: "
                                "each leaf filled with its mean, teal outline."))),
    Algorithm("region_watershed", "Watershed", "segmentation",
              "Slides 35–36 / lab: the image is a topographic surface — valleys (regional minima) are basins. "
              "Water rises from every minimum (priority flooding, 4-neighbours, levels 0 → 255); where water "
              "from two basins meets, the watershed line is the region boundary. Every regional minimum, also "
              "one caused by noise, becomes a region: over-segmentation. Smoothing (σ) removes small minima.",
              (ChoiceParam("surface", "Surface", (("gradient", "Gradient"), ("intensity", "Intensity"),
                                                  ("inverted", "Inverted")), "gradient",
                           help="GRADIENT: |∇f| (Sobel), basins = flat areas, lines on edges.  INTENSITY: dark "
                                "areas are basins.  INVERTED: bright objects are basins."),
               FloatParam("sigma", "Smoothing σ · 0 = none", 3.0, 0.0, 6.0, step=0.5, decimals=1),
               ChoiceParam("display", "Display", (("overlay", "Overlay"), ("labels", "Labels"),
                                                  ("boundaries", "Watershed lines"), ("surface", "Surface"),
                                                  ("minima", "Regional minima")), "overlay",
                           help="OVERLAY: basins in colour, watershed lines white.  MINIMA: the markers "
                                "flooding starts from."))),
    Algorithm("region_marker_watershed", "Marker-Controlled Watershed", "segmentation",
              "Slides 37–38, steps as in the lab (OpenCV coins pipeline): (1) gradient surface, (2) internal "
              "markers — Otsu threshold, opening, distance transform, dist > α · max, one 8-connected marker per "
              "object, (3) external marker — outside the sure background (dilated opening), (4) watershed "
              "flooding only from the markers, (5) one region per object; background = label 0. Pixels between "
              "the markers are 'unknown' and are decided by the flooding.",
              (_SEG_FOREGROUND,
               IntParam("open_iter", "Opening · 3×3 iterations", 2, 0, 5,
                        help="Erode k times, then dilate k times (3×3 square). Removes noise from the Otsu mask."),
               IntParam("bg_iter", "Sure background · dilations", 3, 0, 10,
                        help="The opening dilated this many times; everything outside is the external marker."),
               FloatParam("alpha", "Internal · dist > α · max", 0.5, 0.1, 0.9, step=0.05, decimals=2,
                          help="Distance-transform threshold for the internal (object) markers."),
               ChoiceParam("surface", "Surface", (("gradient", "Gradient · σ = 1"),
                                                  ("distance", "Inverted distance")), "gradient",
                           help="GRADIENT (slide 37, step 1): Sobel |∇f| after Gaussian σ = 1.  INVERTED "
                                "DISTANCE (lab default): 255 − dist / max · 255, splits touching round objects."),
               ChoiceParam("display", "Display", (("overlay", "Overlay"), ("labels", "Labels"),
                                                  ("boundaries", "Watershed lines"), ("markers", "Markers"),
                                                  ("binary", "Otsu mask"), ("sure_bg", "Sure background"),
                                                  ("distance", "Distance"), ("surface", "Surface")), "overlay",
                           help="MARKERS: internal in colour, external dark, unknown gray."))),
    Algorithm("kmeans", "K-Means Clustering", "segmentation", "Groups pixels into k color clusters.",
              (IntParam("k", "Clusters k", 4, 2, 16), IntParam("attempts", "Attempts", 3, 1, 10))),
    # 06 — Color Processing
    Algorithm("color_space", "Color Space Conversion", "color",
              "Slides 24–41 / lab: converts RGB to a color model — RGB, CMY (C = 1 − R, slide 29), CMYK "
              "(K = min(C, M, Y), C′ = (C − K)/(1 − K)), HSV, HSL, HSI (slides 34–35), CIELAB (sRGB → XYZ → "
              "D65) or YCbCr (OpenCV BT.601). Images are BGR in OpenCV; the conversion uses R, G, B. CHANNELS: "
              "all channels side by side as gray images (H: H/360·255, a*/b*: +128, inks: 255·(1 − v)). "
              "ROUND TRIP: RGB → model → RGB, to compare with A; 8-BIT: through the 8-bit channel images "
              "(quantization error).",
              (ChoiceParam("model", "Color model", _COLOR_MODELS, "hsv"),
               ChoiceParam("output", "Output", (("channels", "Channels"), ("roundtrip", "Round trip"),
                                                ("roundtrip8", "8-bit trip")), "channels",
                           help="CHANNELS: output is wider than the input (channels + 3 px gaps).  ROUND TRIP: "
                                "the reconstructed RGB image; the status bar gives the error.  8-BIT TRIP: the "
                                "round trip through the 8-bit channel images."))),
    Algorithm("channel", "Channel Visualization", "color",
              "One channel of a color model as a full-size image, mapped to 0–255 for display (lab rules): "
              "H 0–360° → H/360·255 (red at both ends), S, V, L, I ×255, L* ×255/100, a*, b* + 128 (128 = "
              "neutral), C, M, Y, K as printed ink 255·(1 − v), R, G, B, Y, Cb, Cr as they are. Gray pixels have "
              "no hue; H = 0° there.",
              (ChoiceParam("model", "Color model", _COLOR_MODELS, "hsv"),
               ChoiceParam("channel", "Channel", (("1", "1 · R C H L* Y"), ("2", "2 · G M S a* Cb"),
                                                  ("3", "3 · B Y V/L/I b* Cr"), ("4", "4 · K (CMYK)")), "1",
                           help="The list follows the color model (e.g. HSV: H, S, V; CMYK: C, M, Y, K).",
                           depends_on="model", option_sets=_CHANNEL_SETS),
               ChoiceParam("style", "Style", (("gray", "Gray"), ("tint", "Tinted")), "gray",
                           help="TINTED (RGB, CMY, CMYK only): the channel in its own color — CMY inks on "
                                "white paper, as in the lab."))),
    Algorithm("color_histogram", "Color Histogram", "color",
              "Slide 42 / lab: one histogram per channel, h_c(k) = number of pixels whose channel value falls "
              "in bin k, over the channel's real range (R, G, B, Y, Cb, Cr 0–256, H 0–360°, S, V, L, I, C, M, "
              "Y, K 0–1, L* 0–100, a*, b* −128–128). Shown under COLOR HISTOGRAM; the image is not changed.",
              (ChoiceParam("model", "Color model", _COLOR_MODELS, "rgb"),
               ChoiceParam("bins", "Bins", (("256", "256"), ("64", "64"), ("32", "32"), ("16", "16")), "256"),
               BoolParam("hue_mask", "Hue: chromatic pixels only", False,
                         help="HSV, HSL, HSI: H counts only pixels with S > 0.15 and V > 0.1 (lab hue "
                              "histogram); gray pixels have no hue."))),
    # 06 — pseudo-color (slayt 43–50; laboratuvar Intensity Slicing, Color Coding)
    Algorithm("intensity_slicing", "Intensity Slicing", "color",
              "Slides 47–49 / lab: pseudo-color starts from a gray image. The intensity function is cut by "
              "horizontal slicing planes; every interval V_k between two planes gets its own color: g(x, y) = c_k "
              "if f(x, y) ∈ V_k. 2 COLORS: one plane l_i — f < l_i → c₁, f ≥ l_i → c₂. 4 / 8 COLORS: planes at "
              "round(k · 256 / n) (equal intervals, lab). Color images are first converted to Y. The colors are "
              "artificial; the intervals and colors are listed under COLOR LEGEND.",
              (ChoiceParam("colors", "Colors", (("2", "2 · one plane"), ("4", "4"), ("8", "8")), "2"),
               IntParam("plane", "Plane l_i · 2 colors", 128, 1, 255,
                        help="2 COLORS only: intensities ≥ l_i take c₂."),
               ChoiceParam("palette", "Slice colors", (("lab", "Lab palette"),) + _COLORMAP_CHOICES, "lab",
                           help="LAB: the lab's colors (blue / yellow for 2).  A color map (AYLAB option): each "
                                "slice takes the map's color at the middle of its interval."))),
    Algorithm("pseudocolor", "Color Coding", "color",
              "Slides 44, 50 / lab: color coding with a color map — three independent transfer functions "
              "R = f_R(f), G = f_G(f), B = f_B(f), stored as a 256-entry look-up table: g(x, y) = LUT[f(x, y)]. "
              "Maps as defined in the lab (piecewise linear stops; HSV: hue 0–300°). They approximate, but do "
              "not equal, MATLAB / OpenCV maps. Color images are first converted to Y; the color bar is shown "
              "under COLOR LEGEND.",
              (ChoiceParam("cmap", "Color map", _COLORMAP_CHOICES, "jet"),)),
    # 06 — full-color (slayt 51–56; laboratuvar ctrans, cslice, tone, balance, histeq)
    Algorithm("color_transform", "Color Transformation", "color",
              "Slides 51–52 (Fig. 6.29) / lab: s_i = T_i(r_1, …, r_n). The same effect — reducing the intensity "
              "by a factor k — written in different color models: RGB s_i = k·r_i; CMY s_i = k·r_i + (1 − k); "
              "CMYK (slide figure, not in the lab) only K: s_4 = k·r_4 + (1 − k); HSI only I: s_3 = k·r_3, H and S "
              "unchanged. The formulas differ, the result does not: the report gives the largest difference from "
              "the other models (rounding only). Transfer functions are drawn under TRANSFER CURVES.",
              (ChoiceParam("model", "Color model", (("rgb", "RGB · s_i = k·r_i"), ("cmy", "CMY · s_i = k·r_i + (1 − k)"),
                                                    ("cmyk", "CMYK · s_4 = k·r_4 + (1 − k)"),
                                                    ("hsi", "HSI · s_3 = k·r_3")), "rgb",
                           help="Switch the model with the same k and compare the results (A/B, Undo)."),
               FloatParam("k", "k · intensity factor", 0.7, *color_processing.TRANSFORM_K_RANGE, step=0.05,
                          decimals=2, help="Lab: 0.1 … 1; Fig. 6.29 uses k = 0.7 (30 % darker)."))),
    Algorithm("color_slicing", "Color Slicing", "color",
              "Slide 53 / lab: highlight a range of colors. Pixels whose color lies inside a cube of width W (or a "
              "sphere of radius R₀) centred at the prototype color a keep their color; all others become neutral "
              "gray (0.5, 0.5, 0.5) = 128. CUBE: s_i = 0.5 if |r_j − a_j| > W/2 for any j, else r_i. SPHERE: "
              "s_i = 0.5 if Σ(r_j − a_j)² > R₀², else r_i. Click the image to pick a (from B, the input of Apply).",
              (ChoiceParam("shape", "Region", (("cube", "Cube · W"), ("sphere", "Sphere · R₀")), "cube"),
               IntParam("width", "Cube · width W", 110, *color_processing.SLICE_SIZE_RANGE,
                        help="CUBE only: keep |r_j − a_j| ≤ W/2 in R, G and B."),
               IntParam("radius", "Sphere · radius R₀", 80, *color_processing.SLICE_SIZE_RANGE,
                        help="SPHERE only: keep the Euclidean RGB distance to a ≤ R₀."),
               IntParam("proto_r", "Prototype a · R", 190, 0, 255, help="Click the image to pick a; lab default "
                                                                         "a = (190, 25, 45), a strawberry red."),
               IntParam("proto_g", "Prototype a · G", 25, 0, 255),
               IntParam("proto_b", "Prototype a · B", 45, 0, 255))),
    Algorithm("tone_correction", "Tone & Color Corrections", "color",
              "Slide 54 (Fig. 6.33) / lab: tonal corrections fix contrast and brightness without changing the "
              "colors — the same transfer function is applied to R, G and B: s_i = T(r_i). FLAT (low contrast) → "
              "S-curve T = (s(r) − s(0)) / (s(1) − s(0)), s(r) = 1 / (1 + e^(−a(r − 0.5))); too LIGHT → power law "
              "T(r) = r^γ with γ > 1; too DARK → γ < 1. Lab cases: flat a = 9, light γ = 2, dark γ = 0.45.",
              (ChoiceParam("curve", "Curve", (("s", "S-curve"), ("gamma", "Power · r^γ")), "s"),
               FloatParam("strength", "S-curve · strength a", 9.0, *color_processing.TONE_STRENGTH_RANGE, step=0.5,
                          decimals=1, help="S-CURVE only: steeper mid-tones for larger a (lab 1 … 16)."),
               FloatParam("gamma", "Power · γ", 1.0, *color_processing.TONE_GAMMA_RANGE, step=0.05, decimals=2,
                          help="POWER only: γ > 1 darkens (light images), γ < 1 brightens (dark images)."))),
    Algorithm("color_balance", "Color Balancing", "color",
              "Slide 55 (Fig. 6.36) / lab: a color cast shifts all colors towards one hue; each channel is scaled "
              "(or curved) separately so that neutral objects become neutral again — white balance. "
              "R′ = k_R·R, G′ = k_G·G, B′ = k_B·B. WHITE PATCH: k_c = 255 / max_c. GREY WORLD: k_c = m̄ / m_c "
              "(m_c channel mean, m̄ their mean). NEUTRAL PIXEL: click a pixel that should be gray; its channels "
              "become equal. GAINS (AYLAB): k_R, k_G, k_B by hand. CURVES (Fig. 6.36): c′ = 255·(c / 255)^γ_c — "
              "γ > 1 weakens a channel, γ < 1 strengthens it.",
              (ChoiceParam("method", "Method", (("white", "White patch · 255 / max"), ("gray", "Grey world · m̄ / m"),
                                                ("neutral", "Neutral pixel · click"), ("gains", "Channel gains · k_c"),
                                                ("curves", "Channel curves · γ_c")), "white"),
               IntParam("neutral_r", "Neutral pixel · R", 128, 0, 255,
                        help="NEUTRAL only. Click the image to pick the pixel (from B); lab default (128, 128, 128)."),
               IntParam("neutral_g", "Neutral pixel · G", 128, 0, 255),
               IntParam("neutral_b", "Neutral pixel · B", 128, 0, 255),
               FloatParam("gain_r", "Gains · k_R", 1.0, *color_processing.BALANCE_GAIN_RANGE, step=0.01, decimals=2,
                          help="GAINS only (AYLAB addition): R′ = k_R·R, clipped to 255."),
               FloatParam("gain_g", "Gains · k_G", 1.0, *color_processing.BALANCE_GAIN_RANGE, step=0.01, decimals=2),
               FloatParam("gain_b", "Gains · k_B", 1.0, *color_processing.BALANCE_GAIN_RANGE, step=0.01, decimals=2),
               FloatParam("gamma_r", "Curves · γ_R", 1.0, *color_processing.BALANCE_GAMMA_RANGE, step=0.02, decimals=2,
                          help="CURVES only (lab 0.4 … 2.5). Heavy in cyan = too little red: γ_R < 1."),
               FloatParam("gamma_g", "Curves · γ_G", 1.0, *color_processing.BALANCE_GAMMA_RANGE, step=0.02, decimals=2),
               FloatParam("gamma_b", "Curves · γ_B", 1.0, *color_processing.BALANCE_GAMMA_RANGE, step=0.02,
                          decimals=2))),
    Algorithm("hsi_equalization", "Histogram Equalization in HSI", "color",
              "Slide 56 / lab: convert RGB to HSI, equalize only the intensity I and keep hue H and saturation S: "
              "(H, S, I) → (H, S, T(I)) → RGB, T(I) = (L − 1)·Σ_{j ≤ I} p_I(j) on I8 = round(255·I). A uniform "
              "brightness distribution without color shifts (median I → about 0.5). R, G, B SEPARATELY is the "
              "lab's comparison: three different curves can shift the hues. Saturation × (lab): S′ = min(1, S·×), "
              "HSI only.",
              (ChoiceParam("method", "Method", (("hsi", "Equalize I only · HSI"), ("rgb", "R, G, B separately")),
                           "hsi", help="HSI is the slide-56 method; R, G, B is shown for comparison only."),
               FloatParam("saturation", "Saturation ×", 1.0, *color_processing.EQ_SAT_RANGE, step=0.05, decimals=2,
                          help="HSI only (lab 0.5 … 1.6): optional saturation boost after equalizing I."))),
    # 06 — renkli yumuşatma / keskinleştirme (slayt 57–58; laboratuvar smooth)
    Algorithm("color_smooth_sharpen", "Color Smoothing & Sharpening", "color",
              "Slides 57–58 (Figs. 6.38–6.39) / lab: neighbourhood operations on color vectors. SMOOTHING: k×k "
              "average c̄(x, y) = [(1/K) Σ R(s, t), (1/K) Σ G(s, t), (1/K) Σ B(s, t)]. SHARPENING: Laplacian "
              "g = f − ∇²f = f + (4f − 4 neighbours), kernel [0 −1 0; −1 4 −1; 0 −1 0], always 3×3. EACH RGB "
              "CHANNEL filters R, G and B separately; ONLY I OF HSI filters the intensity and keeps H and S "
              "(I′ clipped to [0, 1]). Borders are replicated (lab). DIFFERENCE × 8 shows c8(8·max_c |RGB_c − HSI_c|) "
              "between the two methods — non-zero only along color edges.",
              (ChoiceParam("operation", "Operation · average / Laplacian", (("smooth", "Smoothing"),
                                                                            ("sharpen", "Sharpening")), "smooth",
                           help="SMOOTHING: k×k average.  SHARPENING: f − ∇²f with the 3×3 Laplacian."),
               ChoiceParam("kernel", "Kernel · smoothing", (("3", "3×3"), ("5", "5×5 · Fig. 6.38")), "3",
                           help="SMOOTHING only. Sharpening always uses the 3×3 Laplacian (lab)."),
               ChoiceParam("process", "Process", (("rgb", "Each RGB channel"), ("hsi", "Only I of HSI")), "rgb"),
               ChoiceParam("display", "Display", (("result", "Result"), ("difference", "|RGB − HSI| × 8")), "result",
                           help="RESULT: the chosen process.  |RGB − HSI| × 8: difference of the two processes "
                                "(gray), as in Figs. 6.38c / 6.39c."))),
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
    "erode": morphology.erode,
    "dilate": morphology.dilate,
    "opening": morphology.opening,
    "closing": morphology.closing,
    "gradient": morphology.gradient,
    "hit_or_miss": morphology.hit_or_miss,
    "top_hat": morphology.top_hat,
    "black_hat": morphology.black_hat,
    "hat_correction": morphology.hat_correction,
    "boundary": morphology.boundary_extraction,
    "hole_filling": morphology.hole_filling,
    "connected_components": morphology.connected_components,
    "convex_hull": morphology.convex_hull,
    "thinning": morphology.thinning,
    "thickening": morphology.thickening,
    "skeletonization": morphology.skeletonization,
    "pruning": morphology.pruning,
    "gray_erosion": morphology.grayscale_erosion,
    "gray_dilation": morphology.grayscale_dilation,
    "gray_opening": morphology.grayscale_opening,
    "gray_closing": morphology.grayscale_closing,
    "gray_gradient": morphology.grayscale_gradient,
    "threshold": segmentation.manual_threshold,
    "iterative_threshold": segmentation.automatic_threshold,
    "otsu": segmentation.otsu_threshold,
    "adaptive_mean": segmentation.adaptive_mean_threshold,
    "adaptive_gaussian": segmentation.adaptive_gaussian_threshold,
    "moving_average": segmentation.moving_average_threshold,
    "niblack": segmentation.niblack_threshold,
    "sauvola": segmentation.sauvola_threshold,
    "edge_discontinuity": segmentation.discontinuity,
    "edge_point": segmentation.point_detection,
    "edge_line": segmentation.line_detection,
    "edge_gradient": segmentation.gradient,
    "edge_roberts": segmentation.roberts,
    "edge_sobel": segmentation.sobel,
    "edge_prewitt": segmentation.prewitt,
    "edge_laplacian": segmentation.laplacian_edges,
    "edge_log": segmentation.laplacian_of_gaussian,
    "edge_canny": segmentation.canny,
    "edge_linking": segmentation.edge_linking,
    "edge_boundary": segmentation.boundary_extraction,
    "edge_hough": segmentation.hough_lines,
    "region_growing": segmentation.region_growing,
    "region_split_merge": segmentation.split_and_merge,
    "region_watershed": segmentation.watershed,
    "region_marker_watershed": segmentation.marker_controlled_watershed,
    "color_space": color_processing.color_space_conversion,
    "channel": color_processing.channel_visualization,
    "color_histogram": color_processing.color_histogram,
    "intensity_slicing": color_processing.intensity_slicing,
    "pseudocolor": color_processing.color_coding,
    "color_transform": color_processing.color_transformation,
    "color_slicing": color_processing.color_slicing,
    "tone_correction": color_processing.tone_correction,
    "color_balance": color_processing.color_balance,
    "hsi_equalization": color_processing.hsi_equalization,
    "color_smooth_sharpen": color_processing.color_smoothing_sharpening,
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
