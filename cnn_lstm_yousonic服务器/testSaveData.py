import torch

data_dict = torch.load("same_voice_data.pt",map_location="cpu")
print(data_dict["Single_Building_Lobby_1_M7_s4_Babble_0dB"])
for k,datas in data_dict.items():
    print(k,datas)

print(len(data_dict.keys()))