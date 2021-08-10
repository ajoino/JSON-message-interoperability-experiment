from json2vec import JSONTreeLSTM
import torch
from datasets import load_seismic_dataset
from pytorch_lightning_test import JsonTreeSystem
from json_decoder_lstm import JSONStructureDecoder

jsons, vectors, labels = load_seismic_dataset()

num_classes = 1
embedder = JsonTreeSystem(mem_dim=64)
output_layer = torch.nn.Linear(128, num_classes)
model = torch.nn.Sequential(embedder)

some_json = r"""
[{"n": "OO_temp_sensor", "t": 0, "u": "K", "v": 290.02483570765054},
{"n": "CC_temp_sensor", "t": 0, "u": "K", "v": 290.032384426905},
{"n": "NW_temp_sensor", "t": 0, "u": "K", "v": 289.98829233126384},
{"n": "NW_Heater", "t": 0, "u": "W", "v": 185.8732269977827},
{"n": "NN_temp_sensor", "t": 0, "u": "K", "v": 290.0789606407754},
{"n": "NN_Heater", "t": 0, "u": "W", "v": 171.3662974759336},
{"n": "NE_temp_sensor", "t": {"1": 1}, "u": "K", "v": 289.97652628070324}
]
"""
same_json = """
{"n": "OO_temp_sensor", "u": "K", "v": 290.02483570765054, "t": 0}
"""
"""
from tqdm import tqdm
for some_json in tqdm(jsons):
    output_3 = model(some_json)
"""
#model("""[1, [2, {"num": {"as_text": "four", "as_int": 4}}]]""")
from prettytable import PrettyTable
from json_data_module import JSONDataModule

data_module = JSONDataModule("some_json.json"); data_module.setup()
data_loader = data_module.train_dataloader()

json_decoder = JSONStructureDecoder(128)

output = json_decoder(torch.randn(1,10,128), torch.randn(1,10,128))

exit()

for batch in data_loader:
    output = model(batch)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(embedder)