import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config import APPAREIL, DOSSIER_RESULTATS


@torch.no_grad()
def evaluer(modele, chargeur, width_mean, width_std, largeurs_vraies, nom=''):
    modele.eval()
    preds_all = []

    for seqs, largs, metas, longs in chargeur:
        seqs  = seqs.to(APPAREIL)
        metas = metas.to(APPAREIL)
        preds = modele(seqs, longs, metas)
        preds_all.extend(preds.cpu().numpy())

    preds_m = np.array(preds_all) * width_std + width_mean
    mae     = np.mean(np.abs(preds_m - largeurs_vraies))
    rmse    = np.sqrt(np.mean((preds_m - largeurs_vraies) ** 2))

    print(f'comme resultat du [{nom}] MAE: {mae:.2f}m | RMSE: {rmse:.2f}m')
    for i in range(min(10, len(preds_m))):
        print(f'    Predit: {preds_m[i]:6.1f}m | Vrai: {largeurs_vraies[i]:6.1f}m | Err: {abs(preds_m[i]-largeurs_vraies[i]):5.1f}m')

    return mae, rmse


def afficher_courbes(historique):
    plt.figure(figsize=(8, 5))
    plt.plot(historique['perte_train'], label='Train')
    plt.plot(historique['perte_val'],   label='Val')
    plt.title('LSTM Width - Perte')
    plt.xlabel('Epoque')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOSSIER_RESULTATS, 'courbes_lstm_width.png'), dpi=150, bbox_inches='tight')
    plt.close()


def afficher_comparaison(mae_val, mae_reel):
    print()
    print('  COMPARAISON')
    print()
    print(f'  CNN Regression   : MAE = 14.91 m')
    print(f'  Mesure Physique  : MAE =  2.40 m')
    print(f'  LSTM (ce modele) : MAE = {mae_reel:.2f} m')

