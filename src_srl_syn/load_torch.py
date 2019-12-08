import torch
model = torch.load("./multibert.bin",map_location='cpu')
#print(model)
state_dict = model["state_dict"]
for var_name in list(state_dict.keys()):
    name = var_name.replace("bert.", "", 1)
    state_dict[name] = state_dict[var_name]
    state_dict.pop(var_name)
print("good")
torch.save(state_dict,"./mtl_bert.bin")

# model2 = torch.load("C:/Users/xzhzhang/Desktop/project/google-tuned-bert/LARGE-BERT-UNCASED/pytorch_model.bin",map_location='cpu')
# print(model2)
#
# aa = "abcabcabc"
# print(aa.replace("abc","", 1))