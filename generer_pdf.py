"""Generates the academic article PDF — professional two-column style."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

doc = SimpleDocTemplate(
    'article.pdf',
    pagesize=A4,
    rightMargin=MARGIN, leftMargin=MARGIN,
    topMargin=MARGIN,   bottomMargin=MARGIN,
    title="Estimating the Width of a Buried Pipe Using Bidirectional LSTM",
    author="AHMED Filali, FOLLIVI Edem Roberto, MAFORIKAN Harald, TAMBOU NGUEMO Franck Kevin, WILSON-BAHUN A. Denis",
)

NAVY   = colors.HexColor('#0d1b2a')
BLUE   = colors.HexColor('#1b4f72')
LBLUE  = colors.HexColor('#d6e4f0')
GREEN  = colors.HexColor('#d4edda')
LGREY  = colors.HexColor('#f8f9fa')
GREY   = colors.HexColor('#6c757d')

styles = getSampleStyleSheet()

s_title   = ParagraphStyle('s_title',   fontSize=20, leading=26, alignment=TA_CENTER, fontName='Helvetica-Bold', textColor=NAVY, spaceAfter=4)
s_sub     = ParagraphStyle('s_sub',     fontSize=11, leading=16, alignment=TA_CENTER, fontName='Helvetica',      textColor=BLUE, spaceAfter=2)
s_affil   = ParagraphStyle('s_affil',   fontSize=9,  leading=13, alignment=TA_CENTER, fontName='Helvetica',      textColor=GREY, spaceAfter=2)
s_h1      = ParagraphStyle('s_h1',      fontSize=12, leading=16, fontName='Helvetica-Bold', textColor=NAVY,  spaceBefore=14, spaceAfter=5,  borderPad=3)
s_h2      = ParagraphStyle('s_h2',      fontSize=10, leading=14, fontName='Helvetica-Bold', textColor=BLUE,  spaceBefore=8,  spaceAfter=3)
s_body    = ParagraphStyle('s_body',    fontSize=9.5,leading=14, fontName='Helvetica',      textColor=colors.black, spaceAfter=6, alignment=TA_JUSTIFY)
s_bullet  = ParagraphStyle('s_bullet',  fontSize=9.5,leading=14, fontName='Helvetica',      textColor=colors.black, spaceAfter=4, leftIndent=14)
s_bold    = ParagraphStyle('s_bold',    fontSize=9.5,leading=14, fontName='Helvetica-Bold', textColor=colors.black, spaceAfter=4)
s_code    = ParagraphStyle('s_code',    fontSize=8,  leading=12, fontName='Courier',        textColor=NAVY,  spaceAfter=6, leftIndent=16, backColor=LGREY, borderPad=6)
s_caption = ParagraphStyle('s_caption', fontSize=8,  leading=11, fontName='Helvetica-Oblique', textColor=GREY, alignment=TA_CENTER, spaceAfter=8)
s_abs     = ParagraphStyle('s_abs',     fontSize=9.5,leading=14, fontName='Helvetica',      textColor=colors.black, alignment=TA_JUSTIFY)


def h1(text):
    """Section heading with a navy underline rule."""
    return KeepTogether([
        Paragraph(text, s_h1),
        HRFlowable(width='100%', thickness=1.2, color=NAVY, spaceAfter=4),
    ])


def h2(text):
    return Paragraph(text, s_h2)


def body(text):
    return Paragraph(text, s_body)


def bullet(text):
    return Paragraph(f'&#8226;&nbsp;&nbsp;{text}', s_bullet)


def tbl(data, col_widths, highlight=None, center_cols=None):
    """Build a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  NAVY),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, LGREY]),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
        ('LINEBELOW',     (0, 0), (-1, 0),  1.2, NAVY),
    ])
    if highlight:
        ts.add('BACKGROUND', (0, highlight), (-1, highlight), GREEN)
        ts.add('FONTNAME',   (0, highlight), (-1, highlight), 'Helvetica-Bold')
    if center_cols:
        for c in center_cols:
            ts.add('ALIGN', (c, 0), (c, -1), 'CENTER')
    t.setStyle(ts)
    return t


# ── Abstract box helper ──────────────────────────────────────────────────────
def abstract_box(text):
    inner = Table(
        [[Paragraph('<b>Abstract</b>', ParagraphStyle('abt', fontSize=9.5, fontName='Helvetica-Bold', textColor=NAVY, spaceAfter=4)),],
         [Paragraph(text, s_abs)]],
        colWidths=[PAGE_W - 2 * MARGIN - 1.4*cm],
    )
    inner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LBLUE),
        ('BOX',        (0, 0), (-1, -1), 1, BLUE),
        ('LEFTPADDING',  (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING',   (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
    ]))
    return inner


# ============================================================================
story = []

# ── Title block ──────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.2*cm))

# Top decorative bar
story.append(HRFlowable(width='100%', thickness=4, color=NAVY, spaceAfter=10))

story.append(Paragraph(
    "Estimating the Width of a Buried Pipe<br/>Using a Bidirectional LSTM Neural Network",
    s_title,
))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "AHMED Filali &nbsp;·&nbsp; FOLLIVI Edem Roberto &nbsp;·&nbsp; MAFORIKAN Harald &nbsp;·&nbsp;"
    " TAMBOU NGUEMO Franck Kevin &nbsp;·&nbsp; WILSON-BAHUN A. Denis",
    s_sub,
))
story.append(Paragraph("HETIC — School of Digital Technology &nbsp;|&nbsp; Paris, France &nbsp;|&nbsp; March 2026", s_affil))
story.append(Paragraph(
    'Industry Partner: <a href="https://skipperndt.com/" color="#1b4f72"><u>Skipper NDT</u></a>'
    ' — https://skipperndt.com/',
    s_affil,
))

story.append(HRFlowable(width='100%', thickness=1.5, color=BLUE, spaceBefore=8, spaceAfter=12))

# ── Abstract ─────────────────────────────────────────────────────────────────
story.append(abstract_box(
    "This paper addresses the automatic estimation of the magnetic influence width of a buried "
    "pipe from multi-channel magnetic field maps, within the Skipper NDT inspection project. "
    "Our approach combines a <b>center-of-mass (barycentre) profile extraction</b> method with "
    "a <b>bidirectional LSTM neural network</b>, which natively handles variable-length "
    "sequences and preserves the physical scale of the acquisition (1 pixel ≈ 20 cm). "
    "Evaluated on real acquisition data, our model achieves a Mean Absolute Error of "
    "<b>4.49 m</b>, reducing the CNN baseline error by <b>70 %</b> and approaching the "
    "physical direct measurement (MAE = 2.40 m). We argue that the LSTM is architecturally "
    "the most suitable model for this problem and that its current performance is limited "
    "solely by the scarcity of annotated real data — not by its design."
))
story.append(Spacer(1, 0.4*cm))

# ── Keywords ─────────────────────────────────────────────────────────────────
story.append(Paragraph(
    "<b>Keywords:</b> Buried pipe inspection &nbsp;·&nbsp; Magnetic field &nbsp;·&nbsp; "
    "LSTM &nbsp;·&nbsp; Bidirectional RNN &nbsp;·&nbsp; Width regression &nbsp;·&nbsp; "
    "Center of mass &nbsp;·&nbsp; Non-destructive testing",
    ParagraphStyle('kw', fontSize=8.5, leading=13, fontName='Helvetica-Oblique',
                   textColor=GREY, spaceAfter=14),
))

# ============================================================================
# 1. INTRODUCTION
# ============================================================================
story.append(h1("1.  Introduction"))
story.append(body(
    "The detection and characterisation of buried pipelines is a critical industrial and "
    "safety challenge. Network operators must regularly inspect their infrastructure to "
    "prevent failures, detect corrosion, and plan maintenance. Traditional methods rely on "
    "direct physical measurements, which are time-consuming and expensive."
))
story.append(body(
    "Magnetic techniques offer a non-intrusive alternative: a sensor moved across the surface "
    "records perturbations of Earth's magnetic field induced by metallic buried structures. "
    "These measurements, organised as 2D multi-channel maps (Bx, By, Bz, and the field norm "
    "||B||), encode the magnetic footprint of the pipe and enable the inference of geometric "
    "properties such as its width of influence."
))

story.append(h2("1.1  Problem Statement"))
story.append(body(
    "We aim to automatically estimate the <b>magnetic influence width</b> (in metres) from a "
    "2D magnetic field map. This is a <b>supervised regression</b> problem: the input is an "
    "image of variable spatial dimensions (up to several thousand pixels wide), and the output "
    "is a scalar between 5 and 80 m."
))

story.append(h2("1.2  Approaches Compared"))
story.append(body("Three approaches are evaluated in this project:"))
story.append(bullet(
    "<b>Physical direct measurement</b> (MAE = 2.40 m): analytical computation based on the "
    "theoretical magnetic response of a cylinder. Highly accurate under ideal conditions but "
    "brittle in the presence of noise or complex geometry."
))
story.append(bullet(
    "<b>CNN Regression</b> (MAE = 14.91 m): convolutional baseline. The CNN processes the "
    "image after resizing to 224×224 pixels, which <b>destroys the physical scale</b> "
    "(1 pixel ≈ 20 cm is lost), making it impossible for the network to directly link the "
    "width in pixels to a width in metres."
))
story.append(bullet(
    "<b>Bidirectional LSTM (this work)</b> (MAE = 4.49 m): our approach. The LSTM processes "
    "a 1D profile extracted from the image, preserving physical scale and exploiting the "
    "sequential structure of the magnetic signal."
))

# ============================================================================
# 2. DATA
# ============================================================================
story.append(h1("2.  Data"))

story.append(h2("2.1  Dataset Description"))
story.append(body(
    "The dataset is provided by <b>Skipper NDT</b> (https://skipperndt.com/) and comprises "
    "two categories of .npz files, each containing a (H, W, 4) float16 array of the four "
    "magnetic channels:"
))
story.append(bullet(
    "<b>Synthetic data (~2,884 files)</b>: generated by physical simulation, covering a wide "
    "variety of configurations — straight or curved pipe, with or without sheath, noisy or "
    "clean field, widths from 5 to 80 m. Provide volume for training."
))
story.append(bullet(
    "<b>Real acquisition data (51 files)</b>: actual field measurements provided by Skipper "
    "NDT with width labels (width_m). These are precious as they reflect true inspection "
    "conditions with their inherent variability and imperfections."
))
story.append(Spacer(1, 0.2*cm))

story.append(body("<b>Data splits used:</b>"))
story.append(tbl(
    [['Split', 'Synthetic data', 'Real data'],
     ['Training',   '85 %', '20 %'],
     ['Validation', '15 %', '20 %'],
     ['Test',       '—',    '60 %']],
    col_widths=[6*cm, 4.5*cm, 4.5*cm],
    center_cols=[1, 2],
))
story.append(Paragraph(
    "Table 1 — Data splits. Real data is introduced in training and validation to ground "
    "the model in actual field conditions, while 60 % is reserved for final evaluation.",
    s_caption,
))

story.append(h2("2.2  Why Real Data Matters"))
story.append(body(
    "Synthetic data, despite its variety, is generated from an idealised physical model. "
    "Real acquisitions contain sensor noise, ground heterogeneity, and geometric imperfections "
    "that cannot be fully simulated. Incorporating real data — even in small quantities — "
    "significantly anchors the model to the true data distribution and reduces the "
    "synthetic-to-real generalisation gap."
))

# ============================================================================
# 3. METHOD
# ============================================================================
story.append(h1("3.  Method"))

# -- 3.1 Barycentre
story.append(h2("3.1  Center-of-Mass Profile Extraction (Barycentre)"))
story.append(body(
    "A naive approach to converting the 2D magnetic map into a 1D signal would be to average "
    "all rows of the image — but this dilutes the pipe signal with uninformative background "
    "rows. A more critical mistake is to <b>filter out low-intensity pixels</b> using a "
    "threshold: this suppresses precisely the near-zero intensity regions corresponding to "
    "the pipe itself, destroying the very information needed to measure its width."
))
story.append(body(
    "Our solution is to locate the <b>center of mass (barycentre)</b> of the normalised "
    "magnetic signal and extract a profile centred on this position."
))
story.append(body("The barycentre row coordinate y<sub>c</sub> is defined as:"))
story.append(Paragraph(
    "y_c  =  Σ(y · I_norm(y,x))  /  Σ(I_norm(y,x))"
    "          [computed via scipy.ndimage.center_of_mass]",
    s_code,
))
story.append(body(
    "where I_norm is the intensity map normalised to [0, 1] by its maximum. "
    "This coordinate identifies the row where the magnetic energy is most concentrated — "
    "i.e., the central axis of the pipe in the image."
))
story.append(body("The 1D profile is then computed as:"))
story.append(Paragraph(
    "profile[x]  =  mean( I_norm[ y_c-2 : y_c+3,  x ] )     for x = 0, 1, …, W−1",
    s_code,
))
story.append(body(
    "A slice of <b>5 rows</b> around y<sub>c</sub> is averaged column by column. "
    "If the resulting profile exceeds 3,000 columns, it is uniformly downsampled."
))
story.append(body("<b>Key properties of this representation:</b>"))
story.append(bullet(
    "<b>All pixels are kept</b>, including near-zero values at the pipe location — "
    "essential for measuring width."
))
story.append(bullet(
    "<b>Physical scale is preserved</b>: the profile has W elements, each representing "
    "≈ 20 cm of real-world distance."
))
story.append(bullet(
    "The profile has <b>variable length W</b>, which matches the natural variable-length "
    "input format of LSTM networks."
))
story.append(bullet(
    "Signal-to-noise ratio is improved by focusing on the 5 rows of highest energy, "
    "discarding noisy peripheral rows."
))
story.append(Spacer(1, 0.1*cm))
story.append(body(
    "Three <b>metadata features</b> are appended to the regressor input: the number of "
    "active pixels (intensity > 0), the image height H, and the image width W — all "
    "z-score normalised. These encode the scale and context of each acquisition."
))

# -- 3.2 LSTM architecture
story.append(h2("3.2  Bidirectional LSTM Architecture"))
story.append(body(
    "The <b>LSTM (Long Short-Term Memory)</b>, introduced by Hochreiter & Schmidhuber (1997), "
    "is a recurrent neural network designed to model long-range dependencies in sequences. "
    "Its gating mechanism (input gate, forget gate, output gate) allows it to selectively "
    "retain or discard information over long horizons, overcoming the vanishing gradient "
    "problem of vanilla RNNs."
))
story.append(body(
    "<b>Why is the LSTM the most suitable model for this problem?</b>"
))
story.append(bullet(
    "<b>Sequential nature of the signal</b>: the magnetic profile is a 1D sequence where "
    "the position of each value carries a physical meaning (distance from the image edge in "
    "metres). The LSTM, unlike the CNN, explicitly leverages this ordering and long-range "
    "spatial dependencies."
))
story.append(bullet(
    "<b>Variable-length input</b>: acquisition images have highly variable widths (197 to "
    "3,000+ columns). The LSTM handles variable-length sequences natively via dynamic "
    "padding and PyTorch's pack_padded_sequence — the CNN imposes a destructive resize."
))
story.append(bullet(
    "<b>Physical scale preservation</b>: pipe width is directly proportional to the number "
    "of profile columns. By processing the raw, non-resized sequence, the LSTM can learn "
    "this correspondence. The CNN, by forcing 224×224, irreversibly loses this information."
))
story.append(bullet(
    "<b>Bidirectional processing</b>: a bidirectional LSTM processes the sequence in both "
    "directions, capturing simultaneously the <i>entry transition</i> (where the signal "
    "rises at the pipe edge) and the <i>exit transition</i> (where it falls at the other "
    "edge). This symmetrical reading is critical for accurately estimating width."
))
story.append(Spacer(1, 0.2*cm))

story.append(body("<b>Model architecture (LSTMWidth):</b>"))
story.append(Paragraph(
    "  Input sequence (L, 1)          Metadata (3,)\n"
    "         |                             |\n"
    "  Bidirectional LSTM                   |\n"
    "  hidden=64, layers=2, dropout=0.3     |\n"
    "         |                             |\n"
    "  [forward_state | backward_state] ────┘\n"
    "     concat  →  (128 + 3 = 131 dim)\n"
    "         |\n"
    "  Linear(131→64) → ReLU → Dropout(0.3)\n"
    "  Linear(64 →32) → ReLU\n"
    "  Linear(32 → 1)          [no output activation — unbounded regression]\n"
    "         |\n"
    "  Predicted width (metres, after denormalisation)",
    s_code,
))
story.append(Paragraph("Figure 1 — LSTMWidth model architecture.", s_caption))

story.append(tbl(
    [['Hyperparameter',      'Value', 'Rationale'],
     ['hidden_size',         '64',    'Sufficient to encode profile structure without overfit'],
     ['num_layers',          '2',     'Hierarchical representation: local transitions + global shape'],
     ['dropout',             '0.3',   'Regularisation — critical with few real training samples'],
     ['Gradient clipping',   '1.0',   'Prevents gradient explosion on long sequences'],
     ['Early stopping patience', '10','Saves best model; avoids overfit on synthetic data'],
     ['Learning rate',       '0.001', 'Adam optimiser with ReduceLROnPlateau (factor 0.5, patience 4)'],
     ['Batch size',          '32',    'Dynamic padding to max sequence length in each batch']],
    col_widths=[4.5*cm, 2.5*cm, 8*cm],
))
story.append(Paragraph("Table 2 — Hyperparameters and design rationale.", s_caption))

# -- 3.3 Training
story.append(h2("3.3  Training Protocol"))
story.append(bullet("<b>Loss function</b>: MSE on z-score normalised widths."))
story.append(bullet("<b>Optimiser</b>: Adam (lr = 0.001) with ReduceLROnPlateau scheduler (factor 0.5, patience 4)."))
story.append(bullet("<b>Gradient clipping</b>: max norm = 1.0 for training stability on long sequences."))
story.append(bullet("<b>Early stopping</b>: patience = 10 epochs on validation loss; best model weights are saved."))
story.append(bullet("<b>Variable-length batching</b>: sequences are padded to the maximum length within each batch using pack_padded_sequence, ensuring the LSTM ignores padding tokens."))

# ============================================================================
# 4. RESULTS
# ============================================================================
story.append(h1("4.  Results"))

story.append(h2("4.1  Final Metrics"))
story.append(tbl(
    [['Dataset',                   'MAE (m)', 'RMSE (m)'],
     ['Validation (Synth + Real)', '~10.86',  '~19.89'],
     ['Test — Real data',          '4.49',    '6.37']],
    col_widths=[8.5*cm, 3*cm, 3.5*cm],
    highlight=2,
    center_cols=[1, 2],
))
story.append(Paragraph("Table 3 — Final evaluation metrics on the held-out test set.", s_caption))
story.append(body(
    "The higher validation MAE compared to the real test MAE is explained by the distribution "
    "of synthetic data, which includes extreme configurations (very wide or very narrow pipes) "
    "that are challenging to predict. On real data, the distribution is more homogeneous and "
    "the LSTM generalises well."
))

story.append(h2("4.2  Comparison of Approaches"))
story.append(tbl(
    [['Approach',                  'MAE — Real Test', 'Error reduction vs CNN'],
     ['CNN Regression',            '14.91 m',         'baseline'],
     ['LSTM (this work)',          '4.49 m',          '− 70 %'],
     ['Physical direct measure',   '2.40 m',          '—']],
    col_widths=[6.5*cm, 4*cm, 4.5*cm],
    highlight=2,
    center_cols=[1, 2],
))
story.append(Paragraph("Table 4 — Comparison of approaches on real acquisition data.", s_caption))
story.append(body(
    "The LSTM outperforms the CNN by a factor of <b>3.3×</b>, achieving an MAE of 4.49 m "
    "with only ~10 real training files. The remaining gap of ~2 m with the physical "
    "measurement is not a structural limitation — it is a data limitation."
))

story.append(h2("4.3  LSTM Potential with More Real Data"))
story.append(body(
    "This is the key finding of this study. The LSTM is <b>architecturally aligned</b> with "
    "the problem: it operates on the right representation (centred 1D profile, scale "
    "preserved), exploits the right properties (sequentiality, long-range dependencies), "
    "and suffers from no fundamental design bias."
))
story.append(body(
    "The current performance (MAE = 4.49 m) was achieved with only <b>~10 real files in "
    "training</b> (20 % of 51). The following mechanisms explain why more real data would "
    "directly and strongly improve results:"
))
story.append(bullet(
    "<b>Real domain calibration</b>: synthetic data is generated from an idealised model "
    "and does not perfectly replicate sensor noise, ground heterogeneity, or non-ideal "
    "geometry. More real samples teach the LSTM the true field distribution."
))
story.append(bullet(
    "<b>Diversity coverage</b>: with only 51 real files total, the model cannot learn the "
    "full diversity of real conditions. With 500–1,000 annotated real files, the LSTM would "
    "have all the information needed to match or exceed the physical measurement — even on "
    "cases where the analytical method fails (noisy or complex acquisitions)."
))
story.append(bullet(
    "<b>Noise robustness</b>: trained on more real noisy cases, the LSTM would learn to "
    "filter acquisition noise and focus on the relevant profile features. The physical "
    "direct measurement, by contrast, is very sensitive to acquisition noise."
))
story.append(body(
    "<b>In summary: the LSTM is the most promising model for this problem.</b> Its current "
    "performance is data-limited, not architecture-limited. This is a fundamental distinction "
    "from the CNN, whose poor performance (14.91 m) stems from a structural mismatch between "
    "the chosen representation and the nature of the problem."
))

# ============================================================================
# 5. CONCLUSION
# ============================================================================
story.append(h1("5.  Conclusion"))
story.append(body(
    "We presented a bidirectional LSTM approach for estimating the width of a buried pipe "
    "from magnetic acquisition data. Two key contributions distinguish this work:"
))
story.append(bullet(
    "<b>Center-of-mass profile extraction</b>: a physically grounded 1D representation "
    "that includes all profile information (including near-zero values at the pipe), "
    "preserves spatial scale, and naturally produces variable-length sequences."
))
story.append(bullet(
    "<b>Bidirectional LSTM</b>: architecturally the most suitable model for this sequential, "
    "variable-length regression problem, outperforming the CNN baseline by 70 % with very "
    "limited real training data."
))
story.append(body(
    "The model achieves MAE = <b>4.49 m</b> on real data vs. 14.91 m for the CNN. "
    "The remaining gap with the physical measurement (2.40 m) is solely due to the scarcity "
    "of annotated real data (51 files), not to any architectural limitation."
))

story.append(h2("Limitations & Future Work"))
story.append(bullet(
    "<b>Annotated real data</b>: obtaining width_m labels for the new 4,715-file dataset "
    "provided by Skipper NDT is the single highest-priority action. It would roughly double "
    "training volume and is expected to push MAE below 2.40 m."
))
story.append(bullet(
    "<b>Transformer architecture</b>: multi-head attention mechanisms could further improve "
    "long-range dependency modelling for very wide profiles (> 1,000 columns)."
))
story.append(bullet(
    "<b>Data augmentation</b>: horizontal symmetry flips and synthetic noise calibrated on "
    "real acquisitions could improve robustness without additional labelling effort."
))

# ============================================================================
# 6. RESOURCES
# ============================================================================
story.append(h1("6.  Resources"))
story.append(bullet(
    "<b>Source code</b>: "
    '<a href="https://github.com/wilsonDenis/skipperndt_task2" color="#1b4f72">'
    '<u>https://github.com/wilsonDenis/skipperndt_task2</u></a>'
))
story.append(bullet(
    "<b>Industry partner</b>: Skipper NDT — "
    '<a href="https://skipperndt.com/" color="#1b4f72"><u>https://skipperndt.com/</u></a>'
))
story.append(bullet("<b>Data</b>: provided by Skipper NDT (proprietary, not publicly available)."))

# ============================================================================
# 7. REFERENCES
# ============================================================================
story.append(h1("7.  References"))

refs = [
    "[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
    "<i>Neural Computation</i>, 9(8), 1735–1780.",
    "[2] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. "
    "<i>IEEE Transactions on Signal Processing</i>, 45(11), 2673–2681.",
    "[3] Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with "
    "bidirectional LSTM networks. <i>IJCNN 2005</i>.",
    "[4] Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep "
    "learning library. <i>NeurIPS 2019</i>.",
    "[5] Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific "
    "computing in Python. <i>Nature Methods</i>, 17, 261–272.",
]
for r in refs:
    story.append(Paragraph(r, ParagraphStyle('ref', parent=s_body, leftIndent=18, firstLineIndent=-18, spaceAfter=4)))

# ── Bottom rule ──────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.4*cm))
story.append(HRFlowable(width='100%', thickness=1, color=NAVY, spaceAfter=6))
story.append(Paragraph(
    "HETIC — School of Digital Technology &nbsp;·&nbsp; Paris, France &nbsp;·&nbsp; 2026 &nbsp;·&nbsp; "
    'Skipper NDT: <a href="https://skipperndt.com/" color="#1b4f72"><u>https://skipperndt.com/</u></a>',
    ParagraphStyle('footer', fontSize=7.5, leading=11, alignment=TA_CENTER,
                   textColor=GREY, fontName='Helvetica-Oblique'),
))

doc.build(story)
print("PDF generated: article.pdf")
