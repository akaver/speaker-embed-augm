import torch
import ECAPA_TDNN

if __name__ == "__main__":
    positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    val_eer, threshold = ECAPA_TDNN.EER(positive_scores, negative_scores)
    print(val_eer)
