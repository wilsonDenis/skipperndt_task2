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
    title="Estimation de la Largeur d'un Pipe Enfoui par LSTM Bidirectionnel",
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
    "Estimation de la Largeur d'un Pipe Enfoui<br/>par Réseau de Neurones Récurrent LSTM Bidirectionnel",
    s_title,
))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "AHMED Filali &nbsp;·&nbsp; FOLLIVI Edem Roberto &nbsp;·&nbsp; MAFORIKAN Harald &nbsp;·&nbsp;"
    " TAMBOU NGUEMO Franck Kevin &nbsp;·&nbsp; WILSON-BAHUN A. Denis",
    s_sub,
))
story.append(Paragraph("HETIC — École du Numérique &nbsp;|&nbsp; Paris, France &nbsp;|&nbsp; Mars 2026", s_affil))
story.append(Paragraph(
    'Partenaire industriel : <a href="https://skipperndt.com/" color="#1b4f72"><u>Skipper NDT</u></a>'
    ' — https://skipperndt.com/',
    s_affil,
))

story.append(HRFlowable(width='100%', thickness=1.5, color=BLUE, spaceBefore=8, spaceAfter=12))

# ── Résumé ───────────────────────────────────────────────────────────────────
story.append(abstract_box(
    "Cet article traite de l'estimation automatique de la largeur d'influence magnétique d'un "
    "pipe enfoui à partir de cartes de champ magnétique multi-canaux, dans le cadre du projet "
    "d'inspection Skipper NDT. Notre approche combine une méthode d'extraction de profil par "
    "<b>barycentre (centre de masse)</b> et un <b>réseau LSTM bidirectionnel</b>, qui traite "
    "nativement des séquences de longueur variable tout en préservant l'échelle physique de "
    "l'acquisition (1 pixel ≈ 20 cm). Évalué sur des données d'acquisition réelles, notre "
    "modèle atteint une erreur absolue moyenne de <b>4,49 m</b>, réduisant l'erreur du CNN de "
    "référence de <b>70 %</b> et s'approchant de la mesure physique directe (MAE = 2,40 m). "
    "Nous montrons que le LSTM est architecturalement le modèle le plus adapté à ce problème "
    "et que ses performances actuelles sont limitées par la rareté des données réelles annotées "
    "— non par sa conception."
))
story.append(Spacer(1, 0.4*cm))

# ── Mots-clés ────────────────────────────────────────────────────────────────
story.append(Paragraph(
    "<b>Mots-clés :</b> Inspection de pipe enfoui &nbsp;·&nbsp; Champ magnétique &nbsp;·&nbsp; "
    "LSTM &nbsp;·&nbsp; RNN bidirectionnel &nbsp;·&nbsp; Régression de largeur &nbsp;·&nbsp; "
    "Barycentre &nbsp;·&nbsp; Contrôle non destructif",
    ParagraphStyle('kw', fontSize=8.5, leading=13, fontName='Helvetica-Oblique',
                   textColor=GREY, spaceAfter=14),
))

# ============================================================================
# 1. INTRODUCTION
# ============================================================================
story.append(h1("1.  Introduction"))
story.append(body(
    "La détection et la caractérisation de pipelines enfouis constituent un enjeu industriel "
    "et de sécurité critique. Les opérateurs de réseaux (eau, gaz, hydrocarbures) doivent "
    "inspecter régulièrement leurs infrastructures afin de prévenir les ruptures, détecter "
    "les corrosions et planifier les interventions de maintenance. Les méthodes traditionnelles, "
    "reposant sur des mesures physiques directes, sont coûteuses et chronophages."
))
story.append(body(
    "Les techniques magnétiques offrent une alternative non intrusive : un capteur déplacé en "
    "surface enregistre les perturbations du champ magnétique terrestre induites par les "
    "structures métalliques enfouies. Ces mesures, organisées en cartes 2D multi-canaux "
    "(Bx, By, Bz et la norme ||B||), encodent l'empreinte magnétique du pipe et permettent "
    "d'en déduire des propriétés géométriques telles que sa largeur d'influence."
))

story.append(h2("1.1  Problème traité"))
story.append(body(
    "Nous cherchons à estimer automatiquement la <b>largeur de la zone d'influence "
    "magnétique</b> (en mètres) à partir d'une carte magnétique 2D. Il s'agit d'un problème "
    "de <b>régression supervisée</b> : l'entrée est une image de dimensions spatiales "
    "variables (jusqu'à plusieurs milliers de pixels de large), et la sortie est un scalaire "
    "compris entre 5 et 80 m."
))

story.append(h2("1.2  Approches comparées"))
story.append(body("Trois approches sont évaluées dans ce projet :"))
story.append(bullet(
    "<b>Mesure physique directe</b> (MAE = 2,40 m) : calcul analytique basé sur la réponse "
    "magnétique théorique d'un cylindre. Très précise dans des conditions idéales mais "
    "fragile en présence de bruit ou de géométrie complexe."
))
story.append(bullet(
    "<b>Régression CNN</b> (MAE = 14,91 m) : référence convolutive. Le CNN traite l'image "
    "après redimensionnement à 224×224 pixels, ce qui <b>détruit l'échelle physique</b> "
    "(1 pixel ≈ 20 cm est perdu), rendant impossible le lien direct entre largeur en pixels "
    "et largeur en mètres."
))
story.append(bullet(
    "<b>LSTM bidirectionnel (ce travail)</b> (MAE = 4,49 m) : notre approche. Le LSTM "
    "traite un profil 1D extrait de l'image, préservant l'échelle physique et exploitant "
    "la structure séquentielle du signal magnétique."
))

# ============================================================================
# 2. DONNÉES
# ============================================================================
story.append(h1("2.  Données"))

story.append(h2("2.1  Description du jeu de données"))
story.append(body(
    "Le jeu de données est fourni par <b>Skipper NDT</b> "
    '(<a href="https://skipperndt.com/" color="#1b4f72"><u>https://skipperndt.com/</u></a>) '
    "et comprend deux catégories de fichiers .npz, chacun contenant un tableau (H, W, 4) "
    "en float16 représentant les quatre canaux magnétiques :"
))
story.append(bullet(
    "<b>Données synthétiques (~2 884 fichiers)</b> : générées par simulation physique, "
    "couvrant une grande variété de configurations — pipe droit ou courbé, avec ou sans "
    "fourreau, signal bruité ou propre, largeurs de 5 à 80 m."
))
story.append(bullet(
    "<b>Données d'acquisition réelles (51 fichiers)</b> : mesures terrain réelles fournies "
    "par Skipper NDT avec labels de largeur (width_m). Ces données sont précieuses car elles "
    "reflètent les vraies conditions d'inspection avec leur variabilité inhérente."
))
story.append(Spacer(1, 0.2*cm))

story.append(body("<b>Répartition des données :</b>"))
story.append(tbl(
    [['Ensemble',       'Données synthétiques', 'Données réelles'],
     ['Entraînement',   '85 %',                 '20 %'],
     ['Validation',     '15 %',                 '20 %'],
     ['Test',           '—',                    '60 %']],
    col_widths=[5*cm, 5.5*cm, 5.5*cm],
    center_cols=[1, 2],
))
story.append(Paragraph(
    "Tableau 1 — Répartition des données. Les données réelles sont intégrées à l'entraînement "
    "et à la validation pour ancrer le modèle dans les conditions terrain réelles, "
    "tandis que 60 % sont réservés à l'évaluation finale.",
    s_caption,
))

story.append(h2("2.2  Importance des données réelles"))
story.append(body(
    "Les données synthétiques, malgré leur variété, sont générées à partir d'un modèle "
    "physique idéalisé. Les acquisitions réelles contiennent du bruit de capteur, une "
    "hétérogénéité du sol et des imperfections géométriques impossibles à simuler "
    "entièrement. Intégrer des données réelles — même en petite quantité — ancre "
    "significativement le modèle dans la vraie distribution et réduit l'écart "
    "synthétique-vers-réel."
))

# ============================================================================
# 3. MÉTHODE
# ============================================================================
story.append(h1("3.  Méthode"))

# -- 3.1 Barycentre
story.append(h2("3.1  Extraction du profil par barycentre"))
story.append(body(
    "Une approche naïve pour convertir la carte magnétique 2D en signal 1D consisterait à "
    "faire la moyenne de toutes les lignes de l'image — mais cela dilue le signal du pipe "
    "avec des lignes de fond non informatives. Une erreur plus grave encore serait de "
    "<b>filtrer les pixels de faible intensité</b> via un seuil : cela supprime précisément "
    "les zones d'intensité quasi nulle correspondant au pipe lui-même, détruisant "
    "l'information nécessaire à la mesure de sa largeur."
))
story.append(body(
    "Notre solution consiste à localiser le <b>barycentre (centre de masse)</b> du signal "
    "magnétique normalisé, puis à extraire un profil centré sur cette position."
))
story.append(body("La coordonnée ligne du barycentre y<sub>c</sub> est définie par :"))
story.append(Paragraph(
    "y_c  =  Σ(y · I_norm(y,x))  /  Σ(I_norm(y,x))"
    "          [calculé via scipy.ndimage.center_of_mass]",
    s_code,
))
story.append(body(
    "où I_norm est la carte d'intensité normalisée dans [0, 1] par son maximum. "
    "Cette coordonnée identifie la ligne où l'énergie magnétique est la plus concentrée, "
    "c'est-à-dire l'axe central du pipe dans l'image."
))
story.append(body("Le profil 1D est alors calculé comme suit :"))
story.append(Paragraph(
    "profil[x]  =  moyenne( I_norm[ y_c-2 : y_c+3,  x ] )     pour x = 0, 1, …, W−1",
    s_code,
))
story.append(body(
    "Une tranche de <b>5 lignes</b> autour de y<sub>c</sub> est moyennée colonne par "
    "colonne. Si le profil résultant dépasse 3 000 colonnes, il est sous-échantillonné "
    "uniformément."
))
story.append(body("<b>Propriétés clés de cette représentation :</b>"))
story.append(bullet(
    "<b>Tous les pixels sont conservés</b>, y compris les valeurs quasi nulles à "
    "l'emplacement du pipe — essentielles pour mesurer la largeur."
))
story.append(bullet(
    "<b>L'échelle physique est préservée</b> : le profil compte W éléments, chacun "
    "représentant ≈ 20 cm de distance réelle."
))
story.append(bullet(
    "Le profil a une <b>longueur variable W</b>, ce qui correspond naturellement au format "
    "d'entrée à longueur variable des réseaux LSTM."
))
story.append(bullet(
    "Le rapport signal/bruit est amélioré en se concentrant sur les 5 lignes à plus haute "
    "énergie, écartant les lignes périphériques bruitées."
))
story.append(Spacer(1, 0.1*cm))
story.append(body(
    "Trois <b>métadonnées</b> sont ajoutées à l'entrée du régresseur : le nombre de pixels "
    "actifs (intensité > 0), la hauteur H et la largeur W de l'image — toutes normalisées "
    "par z-score. Elles encodent l'échelle et le contexte de chaque acquisition."
))

# -- 3.2 Architecture LSTM
story.append(h2("3.2  Architecture LSTM bidirectionnel"))
story.append(body(
    "Le <b>LSTM (Long Short-Term Memory)</b>, introduit par Hochreiter & Schmidhuber (1997), "
    "est un réseau de neurones récurrent conçu pour modéliser des dépendances à long terme "
    "dans des séquences. Son mécanisme de portes (porte d'entrée, d'oubli, de sortie) lui "
    "permet de retenir ou d'écarter sélectivement des informations sur de longs horizons, "
    "surmontant le problème du gradient évanescent des RNN classiques."
))
story.append(body("<b>Pourquoi le LSTM est-il le modèle le plus adapté à ce problème ?</b>"))
story.append(bullet(
    "<b>Nature séquentielle du signal</b> : le profil magnétique est une séquence 1D où "
    "la position de chaque valeur a une signification physique (distance depuis le bord de "
    "l'image en mètres). Le LSTM, contrairement au CNN, exploite explicitement cet ordre "
    "et ces dépendances spatiales à long terme."
))
story.append(bullet(
    "<b>Entrée à longueur variable</b> : les images d'acquisition ont des largeurs très "
    "variables (197 à 3 000+ colonnes). Le LSTM gère nativement les séquences de longueur "
    "variable via le padding dynamique et pack_padded_sequence — le CNN impose un "
    "redimensionnement destructeur."
))
story.append(bullet(
    "<b>Préservation de l'échelle physique</b> : la largeur du pipe est directement "
    "proportionnelle au nombre de colonnes du profil. En traitant la séquence brute non "
    "redimensionnée, le LSTM peut apprendre cette correspondance. Le CNN, en forçant "
    "224×224, perd irrémédiablement cette information."
))
story.append(bullet(
    "<b>Traitement bidirectionnel</b> : un LSTM bidirectionnel parcourt la séquence dans "
    "les deux sens, capturant simultanément la <i>transition d'entrée</i> (montée du signal "
    "au bord du pipe) et la <i>transition de sortie</i> (descente à l'autre bord). Cette "
    "lecture symétrique est critique pour estimer précisément la largeur."
))
story.append(Spacer(1, 0.2*cm))

story.append(body("<b>Architecture du modèle (LSTMWidth) :</b>"))
story.append(Paragraph(
    "  Séquence d'entrée (L, 1)       Métadonnées (3,)\n"
    "         |                             |\n"
    "  LSTM bidirectionnel                  |\n"
    "  hidden=64, layers=2, dropout=0.3     |\n"
    "         |                             |\n"
    "  [état_avant | état_arrière] ─────────┘\n"
    "     concat  →  (128 + 3 = 131 dim)\n"
    "         |\n"
    "  Linear(131→64) → ReLU → Dropout(0.3)\n"
    "  Linear(64 →32) → ReLU\n"
    "  Linear(32 → 1)          [pas d'activation en sortie — régression non bornée]\n"
    "         |\n"
    "  Largeur prédite (mètres, après dénormalisation)",
    s_code,
))
story.append(Paragraph("Figure 1 — Architecture du modèle LSTMWidth.", s_caption))

story.append(tbl(
    [['Hyperparamètre',              'Valeur', 'Justification'],
     ['hidden_size',                 '64',     'Suffisant pour encoder la structure du profil'],
     ['num_layers',                  '2',      'Repr. hiérarchique : transitions locales + forme globale'],
     ['dropout',                     '0,3',    'Régularisation — critique avec peu de données réelles'],
     ['Gradient clipping',           '1,0',    'Évite l\'explosion du gradient sur longues séquences'],
     ['Patience early stopping',     '10',     'Sauvegarde le meilleur modèle, évite le surapprentissage'],
     ['Taux d\'apprentissage',       '0,001',  'Adam + ReduceLROnPlateau (facteur 0,5, patience 4)'],
     ['Taille de lot (batch size)',  '32',     'Padding dynamique à la longueur max dans chaque lot']],
    col_widths=[5.2*cm, 2*cm, 9.4*cm],
))
story.append(Paragraph("Tableau 2 — Hyperparamètres et justifications.", s_caption))

# -- 3.3 Entraînement
story.append(h2("3.3  Protocole d'entraînement"))
story.append(bullet("<b>Fonction de perte</b> : MSE sur les largeurs normalisées par z-score."))
story.append(bullet("<b>Optimiseur</b> : Adam (lr = 0,001) avec scheduler ReduceLROnPlateau (facteur 0,5, patience 4)."))
story.append(bullet("<b>Gradient clipping</b> : norme maximale = 1,0 pour la stabilité sur longues séquences."))
story.append(bullet("<b>Early stopping</b> : patience = 10 époques sur la perte de validation ; le meilleur état est sauvegardé."))
story.append(bullet("<b>Lots à longueur variable</b> : les séquences sont paddées à la longueur maximale du lot via pack_padded_sequence, garantissant que le LSTM ignore les tokens de padding."))

# ============================================================================
# 4. RÉSULTATS
# ============================================================================
story.append(h1("4.  Résultats"))

story.append(h2("4.1  Métriques finales"))
story.append(tbl(
    [['Jeu de données',                    'MAE (m)', 'RMSE (m)'],
     ['Validation (Synthétique + Réel)',   '~10,86',  '~19,89'],
     ['Test — Données réelles',            '4,49',    '6,37']],
    col_widths=[8.5*cm, 3*cm, 3.5*cm],
    highlight=2,
    center_cols=[1, 2],
))
story.append(Paragraph("Tableau 3 — Métriques finales sur le jeu de test.", s_caption))
story.append(body(
    "La MAE de validation plus élevée s'explique par la distribution des données synthétiques, "
    "qui inclut des configurations extrêmes (très larges ou très étroites) difficiles à "
    "prédire. Sur les données réelles, la distribution est plus homogène et le LSTM "
    "généralise bien."
))

story.append(h2("4.2  Comparaison des approches"))
story.append(tbl(
    [['Approche',                'MAE — Test réel', 'Réduction d\'erreur vs CNN'],
     ['Régression CNN',          '14,91 m',          'référence'],
     ['LSTM (ce travail)',       '4,49 m',            '− 70 %'],
     ['Mesure physique directe', '2,40 m',            '—']],
    col_widths=[6.5*cm, 4*cm, 5*cm],
    highlight=2,
    center_cols=[1, 2],
))
story.append(Paragraph("Tableau 4 — Comparaison des approches sur données réelles.", s_caption))
story.append(body(
    "Le LSTM surpasse le CNN d'un facteur <b>3,3×</b>, atteignant une MAE de 4,49 m avec "
    "seulement ~10 fichiers réels en entraînement. L'écart résiduel de ~2 m avec la "
    "mesure physique n'est pas une limite structurelle — c'est une limite de données."
))

story.append(h2("4.3  Potentiel du LSTM avec davantage de données réelles"))
story.append(body(
    "C'est le résultat central de cette étude. Le LSTM est <b>architecturalement aligné</b> "
    "avec le problème : il opère sur la bonne représentation (profil 1D centré, échelle "
    "préservée), exploite les bonnes propriétés (séquentialité, dépendances à long terme) "
    "et ne souffre d'aucun biais de conception fondamental."
))
story.append(body(
    "Les performances actuelles (MAE = 4,49 m) ont été obtenues avec seulement <b>~10 "
    "fichiers réels en entraînement</b> (20 % de 51). Voici pourquoi davantage de données "
    "réelles amélioreraient directement et fortement les résultats :"
))
story.append(bullet(
    "<b>Calibration du domaine réel</b> : les données synthétiques sont générées à partir "
    "d'un modèle idéalisé et ne répliquent pas parfaitement le bruit de capteur, "
    "l'hétérogénéité du sol ou la géométrie non idéale. Plus d'échantillons réels "
    "apprennent au LSTM la vraie distribution du terrain."
))
story.append(bullet(
    "<b>Couverture de la diversité</b> : avec seulement 51 fichiers réels au total, le "
    "modèle ne peut pas apprendre la pleine diversité des conditions réelles. Avec "
    "500–1 000 fichiers réels annotés, le LSTM aurait toutes les informations nécessaires "
    "pour égaler — voire dépasser — la mesure physique, y compris sur les cas où la méthode "
    "analytique échoue (acquisitions bruitées ou géométrie complexe)."
))
story.append(bullet(
    "<b>Robustesse au bruit</b> : entraîné sur davantage de cas réels bruités, le LSTM "
    "apprendrait à filtrer le bruit d'acquisition et à se concentrer sur les "
    "caractéristiques pertinentes du profil. La mesure physique directe, elle, est très "
    "sensible au bruit d'acquisition."
))
story.append(body(
    "<b>En résumé : le LSTM est le modèle le plus prometteur pour ce problème.</b> "
    "Ses performances actuelles sont limitées par les données, non par son architecture. "
    "C'est une distinction fondamentale avec le CNN, dont les mauvaises performances "
    "(14,91 m) proviennent d'une inadéquation structurelle entre la représentation "
    "choisie et la nature du problème."
))

# ============================================================================
# 5. CONCLUSION
# ============================================================================
story.append(h1("5.  Conclusion"))
story.append(body(
    "Nous avons présenté une approche LSTM bidirectionnel pour l'estimation de la largeur "
    "d'un pipe enfoui à partir de données magnétiques. Deux contributions clés distinguent "
    "ce travail :"
))
story.append(bullet(
    "<b>Extraction de profil par barycentre</b> : une représentation 1D physiquement fondée "
    "qui inclut toute l'information du profil (y compris les valeurs quasi nulles au niveau "
    "du pipe), préserve l'échelle spatiale et produit naturellement des séquences de "
    "longueur variable."
))
story.append(bullet(
    "<b>LSTM bidirectionnel</b> : le modèle architecturalement le plus adapté à ce "
    "problème de régression séquentielle à longueur variable, surpassant le CNN de référence "
    "de 70 % avec très peu de données réelles d'entraînement."
))
story.append(body(
    "Le modèle atteint MAE = <b>4,49 m</b> sur données réelles contre 14,91 m pour le CNN. "
    "L'écart résiduel avec la mesure physique (2,40 m) est uniquement dû à la rareté des "
    "données réelles annotées (51 fichiers), non à une limitation architecturale."
))

story.append(h2("Limites et perspectives"))
story.append(bullet(
    "<b>Données réelles annotées</b> : obtenir les labels width_m pour le nouveau jeu de "
    "4 715 fichiers fourni par Skipper NDT est la priorité absolue. Cela doublerait environ "
    "le volume d'entraînement et devrait permettre d'atteindre une MAE inférieure à 2,40 m."
))
story.append(bullet(
    "<b>Architecture Transformer</b> : les mécanismes d'attention multi-têtes pourraient "
    "améliorer encore la modélisation des dépendances à long terme pour les profils très "
    "larges (> 1 000 colonnes)."
))
story.append(bullet(
    "<b>Augmentation de données</b> : les symétries horizontales et l'injection de bruit "
    "synthétique calibré sur les acquisitions réelles pourraient améliorer la robustesse "
    "sans effort d'annotation supplémentaire."
))

# ============================================================================
# 6. RESSOURCES
# ============================================================================
story.append(h1("6.  Ressources"))
story.append(bullet(
    "<b>Code source</b> : "
    '<a href="https://github.com/wilsonDenis/skipperndt_task2" color="#1b4f72">'
    '<u>https://github.com/wilsonDenis/skipperndt_task2</u></a>'
))
story.append(bullet(
    "<b>Partenaire industriel</b> : Skipper NDT — "
    '<a href="https://skipperndt.com/" color="#1b4f72"><u>https://skipperndt.com/</u></a>'
))
story.append(bullet("<b>Données</b> : fournies par Skipper NDT (propriétaires, non publiques)."))

# ============================================================================
# 7. BIBLIOGRAPHIE
# ============================================================================
story.append(h1("7.  Bibliographie"))

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

# ============================================================================
# 8. AUTEURS
# ============================================================================
story.append(h1("8.  Auteurs"))
story.append(body(
    "Pour toute question relative à ce travail, veuillez contacter les auteurs :"
))
story.append(Spacer(1, 0.2*cm))

authors_data = [
    ['Nom',                          'E-mail'],
    ['AHMED Filali',                  'ahmedfillali905@gmail.com'],
    ['FOLLIVI Edem Roberto',          'robertfollivi49@gmail.com'],
    ['MAFORIKAN Harald',              'haraldmaforikan@gmail.com'],
    ['TAMBOU NGUEMO Franck Kevin',    'ktambou99@gmail.com'],
    ['WILSON-BAHUN A. Denis',         'wilsonvry@gmail.com'],
]
t_authors = Table(authors_data, colWidths=[8*cm, 8*cm])
t_authors.setStyle(TableStyle([
    ('BACKGROUND',    (0, 0), (-1, 0),  NAVY),
    ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
    ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
    ('FONTSIZE',      (0, 0), (-1, -1), 9),
    ('TOPPADDING',    (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('LEFTPADDING',   (0, 0), (-1, -1), 8),
    ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, LGREY]),
    ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
    ('LINEBELOW',     (0, 0), (-1, 0),  1.2, NAVY),
]))
story.append(t_authors)

# ── Pied de page ─────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width='100%', thickness=1, color=NAVY, spaceAfter=6))
story.append(Paragraph(
    "HETIC — École du Numérique &nbsp;·&nbsp; Paris, France &nbsp;·&nbsp; 2026 &nbsp;·&nbsp; "
    'Skipper NDT : <a href="https://skipperndt.com/" color="#1b4f72"><u>https://skipperndt.com/</u></a>',
    ParagraphStyle('footer', fontSize=7.5, leading=11, alignment=TA_CENTER,
                   textColor=GREY, fontName='Helvetica-Oblique'),
))

doc.build(story)
print("PDF généré : article.pdf")
