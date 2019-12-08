import numpy as np
import torch
import pyximport

lm_label_mask = torch.tensor([1,0,1])
prediction_ids = torch.tensor([1,2,3])
lm_label_mask = lm_label_mask.byte()
prediction_ids = torch.masked_select(prediction_ids, lm_label_mask)
input_ids = torch.tensor([4,4,4])
input_ids.masked_scatter_(lm_label_mask, prediction_ids)
print(input_ids)
gg = input_ids.clone()
print(gg)