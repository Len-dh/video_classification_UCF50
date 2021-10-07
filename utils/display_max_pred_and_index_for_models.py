import torch 


def max_pred_and_idx(inputlist):
    max_sublist_elemt = []
    max_sublist_idx = []
    print('func max_pred_and_idx')
    print('inputlist', inputlist)
    for sublist in inputlist:
        print('sublist', sublist)
        # max_s_t = sublist#.clone().detach()
        max_s_t = torch.argmax(sublist)
        # max_s_t = max_s_t.item()
        max_sublist_idx.append(max_s_t.item())
        max_s_e = max(sublist)
        # max_s_e = max_s_e#.clone().detach()
        # max_s_e = max_s_e.item()
        max_sublist_elemt.append(max_s_e.item())
    return max_sublist_elemt, max_sublist_idx