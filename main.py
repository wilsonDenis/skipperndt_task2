import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config       import APPAREIL
from src.donnees      import precomputer_donnees, creer_dataloaders
from src.modele       import LSTMWidth
from src.entrainement import entrainer
from src.evaluation   import evaluer, afficher_courbes, afficher_comparaison


def main():
  
    print('  TACHE 2 : ESTIMATION LARGEUR PAR LSTM')
    

    sequences, larg_norm, larg_brut, metas, est_reel, w_mean, w_std = \
        precomputer_donnees()

    dl_train, dl_val, dl_reel, idx_val, idx_reel = \
        creer_dataloaders(sequences, larg_norm, larg_brut, metas, est_reel)

    modele = LSTMWidth(hidden_size=64, num_layers=2, meta_size=3).to(APPAREIL)
    print(f'  Modele : {sum(p.numel() for p in modele.parameters()):,} params')

    historique = entrainer(modele, dl_train, dl_val)
    afficher_courbes(historique)
    print()
    print('  EVALUATION')
    print()

    mae_val,  _ = evaluer(modele, dl_val,  w_mean, w_std, larg_brut[idx_val],  'Val Synthetique')
    mae_reel, _ = evaluer(modele, dl_reel, w_mean, w_std, larg_brut[idx_reel], 'Test Reel')

    afficher_comparaison(mae_val, mae_reel)


if __name__ == '__main__':
    main()
