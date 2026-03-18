import os
import copy
import torch
import torch.nn as nn

from src.config import (
    APPAREIL, NOMBRE_EPOQUES, TAUX_APPRENTISSAGE, PATIENCE, DOSSIER_RESULTATS,
)


def entrainer(modele, chargeur_train, chargeur_val):
    critere      = nn.MSELoss()
    optimiseur   = torch.optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)
    planificateur = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, factor=0.5, patience=4
    )

    meilleure_perte   = float('inf')
    compteur_patience = 0
    meilleur_etat     = None
    historique        = {'perte_train': [], 'perte_val': []}
    print()
    print(f'---Entrainement ({NOMBRE_EPOQUES} epoques max)')
    print()

    for epoque in range(NOMBRE_EPOQUES):
        modele.train()
        total_perte, total = 0.0, 0

        for seqs, largs, metas, longs in chargeur_train:
            seqs  = seqs.to(APPAREIL)
            largs = largs.to(APPAREIL)
            metas = metas.to(APPAREIL)
            optimiseur.zero_grad()
            preds = modele(seqs, longs, metas)
            perte = critere(preds, largs)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optimiseur.step()
            total_perte += perte.item() * seqs.size(0)
            total       += seqs.size(0)

        perte_train = total_perte / total

        modele.eval()
        total_pv, total_v = 0.0, 0

        with torch.no_grad():
            for seqs, largs, metas, longs in chargeur_val:
                seqs  = seqs.to(APPAREIL)
                largs = largs.to(APPAREIL)
                metas = metas.to(APPAREIL)
                preds = modele(seqs, longs, metas)
                total_pv += critere(preds, largs).item() * seqs.size(0)
                total_v  += seqs.size(0)

        perte_val = total_pv / total_v

        historique['perte_train'].append(perte_train)
        historique['perte_val'].append(perte_val)
        planificateur.step(perte_val)

        print(f'Epoque [{epoque+1:2d}/{NOMBRE_EPOQUES}] | Perte: {perte_train:.6f}/{perte_val:.6f}')

        if perte_val < meilleure_perte:
            meilleure_perte   = perte_val
            compteur_patience = 0
            meilleur_etat     = copy.deepcopy(modele.state_dict())
            print(f'--Meilleur modele (perte val: {perte_val:.6f})')
        else:
            compteur_patience += 1
            if compteur_patience >= PATIENCE:
                print(f'-- Early stopping a l\'epoque {epoque + 1}')
                break

    if meilleur_etat:
        modele.load_state_dict(meilleur_etat)
        torch.save(meilleur_etat, os.path.join(DOSSIER_RESULTATS, 'modele_lstm_width.pth'))

    return historique
