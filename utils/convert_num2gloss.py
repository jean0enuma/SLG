import torch
from utils.phoenix_cleanup import clean_phoenix_2014,clean_phoenix_2014_trans

def convert_num2gloss(target,target_length,class2gloss,dataset="phoenix"):
    """
    Convert a list of numbers to a list of glosses using a provided mapping.

    Args:
        target (torch.tensor): A list of numbers representing glosses.
        class2gloss (dict): A dictionary mapping numbers to glosses.

    Returns:
        list: A list of glosses corresponding to the input numbers.
    """
    sequences = []
    start_idx=0
    for i in range(len(target_length)):
        batch_glosses=[]
        for j in range(target_length[i]):
            batch_glosses.append(class2gloss[target[start_idx+j].item()])
        glosses=" ".join(batch_glosses)
        if dataset=="phoenix":
            glosses=clean_phoenix_2014(glosses)
        elif dataset=="phoenixT":
            glosses=clean_phoenix_2014_trans(glosses)
        else:
            raise ValueError("dataset must be phoenix or phoenixT")
        sequences.append(glosses)
        start_idx=start_idx+target_length[i]
    return sequences