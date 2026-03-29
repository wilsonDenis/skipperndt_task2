"""Script de génération du PDF de l'article académique."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

doc = SimpleDocTemplate(
    'article.pdf',
    pagesize=A4,
    rightMargin=2.5*cm, leftMargin=2.5*cm,
    topMargin=2.5*cm,   bottomMargin=2.5*cm,
)

styles = getSampleStyleSheet()
title_style  = ParagraphStyle('Title',  parent=styles['Title'],    fontSize=15, spaceAfter=6,  leading=22, alignment=TA_CENTER)
author_style = ParagraphStyle('Author', parent=styles['Normal'],   fontSize=11, spaceAfter=4,  alignment=TA_CENTER, textColor=colors.HexColor('#444444'))
h1_style     = ParagraphStyle('H1',     parent=styles['Heading1'], fontSize=13, spaceBefore=18,spaceAfter=6,  textColor=colors.HexColor('#1a1a2e'))
h2_style     = ParagraphStyle('H2',     parent=styles['Heading2'], fontSize=11, spaceBefore=10,spaceAfter=4,  textColor=colors.HexColor('#16213e'))
body_style   = ParagraphStyle('Body',   parent=styles['Normal'],   fontSize=10, leading=15,    spaceAfter=7,  alignment=TA_JUSTIFY)
bold_style   = ParagraphStyle('Bold',   parent=styles['Normal'],   fontSize=10, leading=14,    spaceAfter=4,  fontName='Helvetica-Bold')
code_style   = ParagraphStyle('Code',   parent=styles['Code'],     fontSize=8,  leading=13,    spaceAfter=6,  fontName='Courier', backColor=colors.HexColor('#f5f5f5'), leftIndent=20, borderPad=6)
bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],   fontSize=10, leading=14,    spaceAfter=4,  leftIndent=20)
note_style   = ParagraphStyle('Note',   parent=styles['Normal'],   fontSize=9,  leading=13,    spaceAfter=5,  leftIndent=20, textColor=colors.HexColor('#555555'), fontName='Helvetica-Oblique')


def make_table(data, col_widths, highlight_row=None):
    t = Table(data, colWidths=col_widths)
    style = TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,-1), 9),
        ('ALIGN',         (1, 0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
        ('GRID',          (0, 0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING',    (0, 0), (-1,-1), 5),
        ('BOTTOMPADDING', (0, 0), (-1,-1), 5),
    ])
    if highlight_row is not None:
        style.add('BACKGROUND', (0, highlight_row), (-1, highlight_row), colors.HexColor('#d4edda'))
        style.add('FONTNAME',   (0, highlight_row), (-1, highlight_row), 'Helvetica-Bold')
    t.setStyle(style)
    return t


story = []

# --------------------------------------------------------------------------
# En-tête
# --------------------------------------------------------------------------
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Estimation de la Largeur d'un Pipe Enfoui<br/>par Réseau de Neurones Récurrent LSTM",
    title_style,
))
story.append(Spacer(1, 0.3*cm))

# Tableau des auteurs
authors_data = [
    ['Auteur', 'Email'],
    ['AHMED Filali',              'ahmedfillali905@gmail.com'],
    ['FOLLIVI Edem Roberto',      'robertfollivi49@gmail.com'],
    ['MAFORIKAN Harald',          'haraldmaforikan@gmail.com'],
    ['TAMBOU NGUEMO Franck Kevin','ktambou99@gmail.com'],
    ['WILSON-BAHUN A. Denis',     'wilsonvry@gmail.com'],
]
t_authors = Table(authors_data, colWidths=[8*cm, 6*cm])
t_authors.setStyle(TableStyle([
    ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
    ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
    ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0, 0), (-1,-1), 9),
    ('ALIGN',         (0, 0), (-1,-1), 'LEFT'),
    ('ROWBACKGROUNDS',(0, 1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
    ('GRID',          (0, 0), (-1,-1), 0.5, colors.grey),
    ('TOPPADDING',    (0, 0), (-1,-1), 4),
    ('BOTTOMPADDING', (0, 0), (-1,-1), 4),
    ('LEFTPADDING',   (0, 0), (-1,-1), 8),
]))
story.append(t_authors)
story.append(Paragraph('HETIC — Mars 2026', author_style))
story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#1a1a2e'), spaceAfter=14))

# --------------------------------------------------------------------------
# Résumé
# --------------------------------------------------------------------------
story.append(Paragraph('Résumé', h1_style))
story.append(Paragraph(
    "Ce travail s'inscrit dans le cadre du projet Skipper, qui vise à automatiser l'inspection "
    "de pipelines enfouis à partir de données de champ magnétique. Nous traitons la <b>tâche "
    "d'estimation de la largeur</b> de la zone d'influence magnétique. Notre approche repose "
    "sur un <b>réseau LSTM bidirectionnel</b> alimenté par un profil d'intensité 1D extrait "
    "autour du <b>barycentre</b> du signal magnétique. Nous démontrons que le LSTM surpasse "
    "significativement l'approche CNN de référence, en réduisant l'erreur absolue moyenne de "
    "14,91 m à <b>4,49 m</b> sur données réelles — soit une réduction de <b>70 %</b>. "
    "Nous montrons également que, structurellement, le LSTM est le modèle le mieux adapté "
    "à ce problème, et que ses performances sont directement liées à la quantité de données "
    "réelles disponibles : avec davantage de données terrain, le LSTM a le potentiel "
    "d'atteindre ou de dépasser la mesure physique directe (MAE = 2,40 m).",
    body_style,
))

# --------------------------------------------------------------------------
# 1. Introduction
# --------------------------------------------------------------------------
story.append(Paragraph('1. Introduction', h1_style))
story.append(Paragraph(
    "La détection et la caractérisation de pipelines enfouis constitue un enjeu industriel "
    "et de sécurité critique. Les opérateurs de réseaux (eau, gaz, hydrocarbures) doivent "
    "inspecter régulièrement leurs infrastructures pour prévenir les ruptures, détecter les "
    "corrosions et planifier les interventions de maintenance. Les méthodes traditionnelles, "
    "basées sur des mesures physiques in situ, sont coûteuses et chronophages.",
    body_style,
))
story.append(Paragraph(
    "Les techniques magnétiques offrent une alternative non intrusive prometteuse. Un capteur "
    "déplacé en surface enregistre les perturbations du champ magnétique terrestre induites "
    "par les structures métalliques enfouies. Ces mesures, organisées en cartes 2D "
    "multi-canaux (Bx, By, Bz et la norme ||B||), contiennent l'empreinte magnétique du pipe "
    "et permettent d'en déduire des caractéristiques géométriques comme sa largeur d'influence.",
    body_style,
))
story.append(Paragraph("Problème traité", h2_style))
story.append(Paragraph(
    "Nous cherchons à estimer automatiquement la <b>largeur de la zone d'influence "
    "magnétique</b> (en mètres) à partir d'une carte magnétique 2D. Il s'agit d'un problème "
    "de <b>régression supervisée</b> : l'entrée est une image de dimensions variables "
    "(plusieurs centaines à milliers de pixels), et la sortie est un scalaire entre 5 et 80 m.",
    body_style,
))
story.append(Paragraph("État de l'art et motivation", h2_style))
story.append(Paragraph(
    "Plusieurs approches ont été explorées dans ce projet :",
    body_style,
))
story.append(Paragraph(
    "• <b>Mesure physique directe</b> (MAE = 2,40 m) : exploitation analytique du modèle "
    "théorique de réponse magnétique d'un cylindre. Très précise dans des conditions "
    "idéales, mais fragile face aux acquisitions bruitées ou à géométrie complexe.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>CNN Régression</b> (MAE = 14,91 m) : approche de référence. Le CNN traite l'image "
    "après redimensionnement à 224×224 pixels. Ce redimensionnement est une perte "
    "d'information critique : l'échelle physique (1 pixel ≈ 20 cm) est détruite, et le "
    "réseau ne peut pas relier directement la largeur en pixels à une largeur en mètres.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>LSTM bidirectionnel</b> (MAE = 4,49 m) : notre approche. Le LSTM travaille sur "
    "une séquence 1D de longueur variable, préservant l'échelle physique et exploitant la "
    "structure temporelle du signal magnétique. C'est l'approche la plus cohérente avec "
    "la nature du problème.",
    bullet_style,
))

# --------------------------------------------------------------------------
# 2. Matériel et Méthodes
# --------------------------------------------------------------------------
story.append(Paragraph('2. Matériel et Méthodes', h1_style))

story.append(Paragraph('2.1 Données', h2_style))
story.append(Paragraph(
    "Le jeu de données provient du projet Skipper et comprend deux catégories :",
    body_style,
))
story.append(Paragraph(
    "• <b>Données synthétiques (~2 884 fichiers)</b> : générées par simulation physique, "
    "elles couvrent une grande variété de configurations — pipe droit ou courbé, avec ou "
    "sans fourreau, signal bruité ou propre, largeurs de 5 à 80 m. Chaque fichier est un "
    "tableau NumPy (H, W, 4) en float16.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Données réelles (51 fichiers)</b> : acquisitions terrain réelles fournies par "
    "Skipper NDT, avec labels de largeur (width_m). Ces données sont précieuses car elles "
    "reflètent les conditions réelles d'inspection avec leurs imperfections et leur variabilité.",
    bullet_style,
))
story.append(Spacer(1, 0.2*cm))
story.append(make_table(
    [['Ensemble',      'Synthétiques', 'Réelles'],
     ['Entraînement',  '85 %',         '20 %'],
     ['Validation',    '15 %',         '20 %'],
     ['Test',          '—',            '60 %']],
    [5*cm, 4*cm, 4*cm],
))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "Les données réelles sont intégrées à l'entraînement et à la validation pour ancrer le "
    "modèle dans la réalité du terrain, tout en réservant 60 % d'entre elles pour l'évaluation "
    "finale.",
    body_style,
))

# --------------------------------------------------------------------------
# 2.2 Barycentre
# --------------------------------------------------------------------------
story.append(Paragraph("2.2 Extraction du profil par barycentre", h2_style))
story.append(Paragraph(
    "L'une des contributions clés de ce travail est la méthode d'extraction du profil 1D "
    "depuis la carte magnétique 2D. Une approche naïve consisterait à prendre la moyenne "
    "de toutes les lignes de l'image — mais cela dilue le signal avec des zones sans "
    "information. Une autre erreur courante est de <b>filtrer les pixels d'intensité faible</b> "
    "(en dessous d'un seuil), ce qui supprime précisément les zones à intensité nulle "
    "correspondant au pipe lui-même — détruisant l'information de largeur.",
    body_style,
))
story.append(Paragraph(
    "Notre solution : localiser le <b>barycentre</b> (centre de masse) du signal magnétique "
    "normalisé, puis extraire un profil centré sur cette position.",
    body_style,
))
story.append(Paragraph(
    "Le barycentre d'une image d'intensité I(y, x) est défini par :",
    body_style,
))
story.append(Paragraph(
    "y_c = Σ(y · I(y,x)) / Σ(I(y,x))     [calculé avec scipy.ndimage.center_of_mass]",
    code_style,
))
story.append(Paragraph(
    "Cette coordonnée y_c indique la ligne où le signal magnétique est le plus concentré, "
    "c'est-à-dire la ligne centrale du pipe dans l'image. On extrait ensuite une <b>tranche "
    "de 5 lignes</b> autour de y_c (y_c − 2 à y_c + 2) et on calcule la moyenne colonne "
    "par colonne pour obtenir le profil final :",
    body_style,
))
story.append(Paragraph(
    "profil[x] = moyenne( I[y_c-2 : y_c+3, x] )    pour x = 0, 1, ..., W-1",
    code_style,
))
story.append(Paragraph(
    "<b>Pourquoi cette approche est supérieure ?</b> Le profil ainsi obtenu :",
    bold_style,
))
story.append(Paragraph(
    "• Contient tous les pixels de la largeur, y compris ceux d'intensité nulle ou faible "
    "(zone du pipe), qui sont porteurs d'information sur la largeur.",
    bullet_style,
))
story.append(Paragraph(
    "• Est centré sur le pipe, éliminant les lignes de bruit situées loin du signal.",
    bullet_style,
))
story.append(Paragraph(
    "• Préserve la longueur naturelle W de l'image en colonnes, donc l'<b>échelle physique</b> "
    "(1 colonne ≈ 20 cm selon la résolution d'acquisition).",
    bullet_style,
))
story.append(Paragraph(
    "• Produit une séquence de longueur variable, naturellement adaptée à un traitement par LSTM.",
    bullet_style,
))
story.append(Paragraph(
    "En complément, trois métadonnées normalisées enrichissent le vecteur d'entrée du "
    "régresseur : le nombre de pixels actifs (intensité > 0), la hauteur H et la largeur W "
    "de l'image. Ces features encodent l'échelle et le contexte de l'acquisition.",
    body_style,
))

# --------------------------------------------------------------------------
# 2.3 Architecture LSTM
# --------------------------------------------------------------------------
story.append(Paragraph("2.3 Architecture LSTM bidirectionnel — choix et justification", h2_style))
story.append(Paragraph(
    "Le <b>LSTM (Long Short-Term Memory)</b> est un réseau de neurones récurrent conçu pour "
    "modéliser des dépendances à long terme dans des séquences. Introduit par Hochreiter & "
    "Schmidhuber (1997), il résout le problème du gradient évanescent des RNN classiques "
    "grâce à un mécanisme de portes (gate mechanism) qui contrôle ce qui est mémorisé, "
    "oublié ou transmis à chaque pas de temps.",
    body_style,
))
story.append(Paragraph(
    "<b>Pourquoi le LSTM est le modèle le plus adapté à ce problème ?</b>",
    bold_style,
))
story.append(Paragraph(
    "• <b>Nature séquentielle du signal</b> : le profil magnétique est une séquence 1D où "
    "la position de chaque valeur dans la séquence a une signification physique (distance "
    "en mètres depuis le bord de l'image). Le LSTM, contrairement au CNN, exploite "
    "explicitement cet ordre et ces dépendances spatiales.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Longueur variable</b> : les images d'acquisition ont des largeurs W très variables "
    "(197 à 3 000+ colonnes). Le LSTM traite nativement des séquences de longueurs variables "
    "via le mécanisme de padding et de pack_padded_sequence — le CNN impose un redimensionnement "
    "destructeur.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Préservation de l'échelle physique</b> : la largeur du pipe est directement liée "
    "au nombre de colonnes dans le profil. En travaillant sur la séquence brute (non "
    "redimensionnée), le LSTM peut apprendre cette correspondance. Le CNN, en forçant "
    "224×224, perd complètement cette information.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Capture des transitions</b> : la largeur se lit dans le profil par la position des "
    "transitions montante et descendante du signal. Un réseau <b>bidirectionnel</b> parcourt "
    "la séquence dans les deux sens, capturant simultanément la transition d'entrée "
    "(début du pipe) et de sortie (fin du pipe) — ce qu'un LSTM unidirectionnel ne ferait "
    "qu'imparfaitement.",
    bullet_style,
))
story.append(Paragraph(
    "L'architecture complète est la suivante :",
    body_style,
))
story.append(Paragraph(
    "Séquence (L, 1)          Métadonnées normalisées (3,)\n"
    "      |                           |\n"
    " LSTM bidirectionnel              |\n"
    " hidden_size=64, num_layers=2     |\n"
    " dropout=0.3                      |\n"
    "      |                           |\n"
    " [état_avant | état_arrière] ─────┘\n"
    "      concaténation (128 + 3 = 131 dimensions)\n"
    "      |\n"
    " Linear(131 → 64) → ReLU → Dropout(0.3)\n"
    " Linear(64  → 32) → ReLU\n"
    " Linear(32  →  1)          [sans activation = régression non bornée]\n"
    "      |\n"
    " Largeur prédite (mètres, après dénormalisation)",
    code_style,
))
story.append(Paragraph(
    "<b>Détails des choix d'architecture :</b>",
    bold_style,
))
story.append(Paragraph(
    "• <b>hidden_size = 64</b> : taille de l'état caché par direction. Suffisant pour encoder "
    "la structure du profil sans sur-paramétrage.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>num_layers = 2</b> : deux couches LSTM empilées permettent au réseau de construire "
    "des représentations hiérarchiques (bas niveau : transitions locales ; haut niveau : "
    "structure globale du profil).",
    bullet_style,
))
story.append(Paragraph(
    "• <b>dropout = 0.3</b> : régularisation appliquée entre les couches LSTM et dans le MLP "
    "pour limiter le sur-apprentissage, particulièrement important avec peu de données réelles.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Gradient clipping (norme max = 1,0)</b> : stabilise l'entraînement en évitant "
    "les explosions de gradient, phénomène fréquent avec les RNN sur de longues séquences.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Pas d'activation ReLU sur la couche de sortie</b> : la régression de largeur "
    "nécessite une sortie non bornée. Une ReLU en sortie bloquerait les prédictions négatives "
    "après dénormalisation.",
    bullet_style,
))
story.append(Paragraph(
    "Le modèle compte environ <b>120 000 paramètres entraînables</b>, ce qui le rend léger "
    "et rapide à entraîner même sur CPU.",
    body_style,
))

story.append(Paragraph("2.4 Protocole d'entraînement", h2_style))
story.append(Paragraph("• <b>Fonction de perte</b> : MSE (Mean Squared Error) sur les largeurs normalisées par z-score.", bullet_style))
story.append(Paragraph("• <b>Optimiseur</b> : Adam (lr = 0,001) avec scheduler ReduceLROnPlateau (facteur 0,5, patience 4 époques sans amélioration).", bullet_style))
story.append(Paragraph("• <b>Early stopping</b> : arrêt si la perte de validation ne s'améliore pas pendant 10 époques consécutives. Le meilleur état du modèle est sauvegardé.", bullet_style))
story.append(Paragraph("• <b>Batch size = 32</b> : les séquences de longueurs variables sont paddées dynamiquement à la longueur maximale du lot via pack_padded_sequence.", bullet_style))

# --------------------------------------------------------------------------
# 3. Résultats
# --------------------------------------------------------------------------
story.append(Paragraph('3. Résultats', h1_style))

story.append(Paragraph('3.1 Métriques finales', h2_style))
story.append(Spacer(1, 0.2*cm))
story.append(make_table(
    [['Jeu de données',           'MAE (m)', 'RMSE (m)'],
     ['Validation (Synth + Réel)','~10,86',  '~19,89'],
     ['Test Réel',                '4,49',    '6,37']],
    [7*cm, 3.5*cm, 3.5*cm],
    highlight_row=2,
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "La MAE de validation plus élevée que sur le test réel s'explique par la distribution "
    "des données synthétiques, dont certaines configurations extrêmes (très large ou très "
    "étroit) sont difficiles à prédire. Sur les données réelles, la distribution est plus "
    "homogène et le LSTM généralise bien.",
    body_style,
))

story.append(Paragraph('3.2 Comparaison des approches', h2_style))
story.append(Spacer(1, 0.2*cm))
story.append(make_table(
    [['Approche',             'MAE Test Réel', 'Réduction d\'erreur'],
     ['CNN Régression',       '14,91 m',       'référence'],
     ['LSTM (ce modèle)',     '4,49 m',        '−70 %'],
     ['Mesure Physique',      '2,40 m',        '—']],
    [5.5*cm, 4*cm, 4.5*cm],
    highlight_row=2,
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Le LSTM réduit l'erreur du CNN d'un facteur <b>3,3</b>. L'écart résiduel de ~2 m avec "
    "la mesure physique directe n'est pas une limite structurelle du modèle — c'est une "
    "conséquence directe du <b>manque de données réelles d'entraînement</b>.",
    body_style,
))

story.append(Paragraph("3.3 Potentiel du LSTM avec davantage de données réelles", h2_style))
story.append(Paragraph(
    "C'est le point central de cette analyse. Le LSTM, contrairement au CNN, est "
    "<b>structurellement cohérent</b> avec le problème : il travaille sur la bonne "
    "représentation (profil 1D centré, échelle préservée) et exploite les bonnes propriétés "
    "(séquentialité, dépendances à long terme). Il ne souffre d'aucun biais architectural.",
    body_style,
))
story.append(Paragraph(
    "Les performances actuelles (MAE = 4,49 m) ont été obtenues avec seulement "
    "<b>~10 fichiers réels en entraînement</b> (20 % de 51). Cette contrainte explique "
    "l'essentiel de l'écart avec la mesure physique. Voici pourquoi davantage de données "
    "réelles aurait un impact direct et fort :",
    body_style,
))
story.append(Paragraph(
    "• <b>Calibration du domaine réel</b> : les données synthétiques sont générées par "
    "simulation et ne reflètent pas parfaitement les imperfections des acquisitions terrain "
    "(bruit de capteur, variations de sol, géométrie non idéale). Plus le modèle voit de "
    "données réelles, mieux il apprend cette distribution.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Généralisation</b> : avec 51 fichiers réels en tout (dont seulement ~10 en train), "
    "le modèle ne peut pas apprendre la diversité des conditions réelles. Avec 500 ou 1 000 "
    "fichiers réels annotés, le LSTM aurait toutes les informations nécessaires pour égaler "
    "— voire surpasser — la mesure physique sur des cas complexes où celle-ci échoue.",
    bullet_style,
))
story.append(Paragraph(
    "• <b>Robustesse au bruit</b> : le LSTM, entraîné sur davantage de cas réels bruités, "
    "apprendrait à filtrer le bruit et à se concentrer sur les caractéristiques pertinentes "
    "du profil. La mesure physique directe, elle, est très sensible au bruit d'acquisition.",
    bullet_style,
))
story.append(Paragraph(
    "En résumé : <b>le LSTM est le modèle le plus prometteur pour ce problème</b>. "
    "Ses performances actuelles sont limitées par les données, pas par son architecture. "
    "C'est une différence fondamentale avec le CNN, dont les mauvaises performances "
    "(14,91 m) tiennent à une inadéquation structurelle entre la représentation choisie "
    "et la nature du problème.",
    body_style,
))

# --------------------------------------------------------------------------
# 4. Conclusion
# --------------------------------------------------------------------------
story.append(Paragraph('4. Conclusion', h1_style))
story.append(Paragraph(
    "Nous avons présenté une approche LSTM bidirectionnel pour l'estimation de la largeur "
    "d'un pipe enfoui à partir de données magnétiques. Deux contributions clés distinguent "
    "cette approche :",
    body_style,
))
story.append(Paragraph(
    "1. L'extraction d'un profil 1D par <b>barycentre</b> (centre de masse du signal), qui "
    "fournit une représentation physiquement cohérente de la zone magnétique, incluant les "
    "zones d'intensité nulle et préservant l'échelle spatiale.",
    bullet_style,
))
story.append(Paragraph(
    "2. L'utilisation d'un <b>LSTM bidirectionnel</b>, qui est architecturalement le modèle "
    "le plus adapté à la nature séquentielle et à longueur variable du profil magnétique.",
    bullet_style,
))
story.append(Paragraph(
    "Ces choix permettent une <b>réduction de 70 % de l'erreur</b> par rapport au CNN de "
    "référence, avec seulement ~10 fichiers réels en entraînement.",
    body_style,
))
story.append(Paragraph('<b>Limites et perspectives :</b>', bold_style))
story.append(Paragraph(
    "Le principal frein est le volume de données réelles annotées (51 fichiers). Un nouveau "
    "jeu de 4 715 fichiers synthétiques a été fourni par Skipper NDT, mais sans annotation "
    "width_m. L'obtention de ces annotations représente la priorité absolue pour la suite "
    "du projet.",
    body_style,
))
story.append(Paragraph(
    "• Avec 500+ fichiers réels annotés, le LSTM a le potentiel d'<b>atteindre ou dépasser "
    "la mesure physique directe</b> (MAE < 2,40 m).",
    bullet_style,
))
story.append(Paragraph(
    "• Des architectures Transformer (attention multi-tête) pourraient encore améliorer la "
    "capture de dépendances long-range dans les profils de grande largeur.",
    bullet_style,
))
story.append(Paragraph(
    "• L'augmentation de données (symétries horizontales, injection de bruit synthétique "
    "calibré sur les données réelles) permettrait de mieux couvrir la distribution réelle.",
    bullet_style,
))

# --------------------------------------------------------------------------
# 5. Ressources
# --------------------------------------------------------------------------
story.append(Paragraph('5. Ressources', h1_style))
story.append(Paragraph("• Code source : https://github.com/wilsonDenis/skipperndt_task2", bullet_style))
story.append(Paragraph("• Données : fournies par l'entreprise Skipper NDT (non publiques).", bullet_style))

# --------------------------------------------------------------------------
# 6. Bibliographie
# --------------------------------------------------------------------------
story.append(Paragraph('6. Bibliographie', h1_style))
story.append(Paragraph(
    "[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
    "<i>Neural Computation</i>, 9(8), 1735–1780.",
    bullet_style,
))
story.append(Paragraph(
    "[2] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. "
    "<i>IEEE Transactions on Signal Processing</i>, 45(11), 2673–2681.",
    bullet_style,
))
story.append(Paragraph(
    "[3] Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with "
    "bidirectional LSTM networks. <i>IJCNN 2005</i>.",
    bullet_style,
))
story.append(Paragraph(
    "[4] Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep "
    "learning library. <i>NeurIPS 2019</i>.",
    bullet_style,
))
story.append(Paragraph(
    "[5] Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific "
    "computing in Python. <i>Nature Methods</i>, 17, 261–272.",
    bullet_style,
))

doc.build(story)
print('PDF genere : article.pdf')
