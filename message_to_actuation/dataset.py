import csv
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class MessageDataset(Dataset):
    def __init__(
            self,
            file_path: Union[Path, str],
            train: bool = False,
            validation: bool = False,
            message_transform: Any = None,
            setpoint_transform: Any = None,
            actuation_transform: Any = None,
            room_transform: Any = None,
    ):
        simulation_data = pd.read_csv(file_path, sep=';').dropna().reset_index(drop=True)
        self.simulation_data = simulation_data[simulation_data['system'] == 'temp_sensor'].reset_index(drop=True)
        self.simulation_data['message'] = self.simulation_data['message'].str.replace("'", '"')
        self.room_categories = pd.Categorical(self.simulation_data['room_name']).categories
        self.simulation_data = self.simulation_data.sample(frac = 1)
        self.train = train
        self.validation = validation
        self.setpoint_transform = setpoint_transform
        self.actuation_transform = actuation_transform
        self.room_transform = room_transform

    def __len__(self) -> int:
        return len(self.simulation_data)

    def __getitem__(self, idx: int) -> Tuple[List[Dict], int, float, float, torch.Tensor]: # Returns message, room label, setpoint, and actuation value
        message = self.simulation_data.loc[idx, 'message']
        room_name_sample, setpoint_sample, actuation_sample, prev_actuation_sample_1, prev_actuation_sample_2, prev_actuation_sample_3 = (
            self.simulation_data.loc[idx, column]
            for column in ('room_name', 'setpoint', 'actuation', 'previous_actuation_1', 'previous_actuation_2', 'previous_actuation_3')
        )

        if self.room_transform:
            room_name_sample = self.room_transform((self.room_categories.get_loc(room_name_sample), ))
        else:
            room_name_sample = self.room_categories(room_name_sample)
        if self.setpoint_transform:
            setpoint_sample = self.setpoint_transform((setpoint_sample, ))
        if self.actuation_transform:
            actuation_sample = self.actuation_transform((actuation_sample, ))
            prev_actuation_sample_1 = self.actuation_transform((prev_actuation_sample_1,))
            prev_actuation_sample_2 = self.actuation_transform((prev_actuation_sample_2,))
            prev_actuation_sample_3 = self.actuation_transform((prev_actuation_sample_3,))
            prev_actuations_sample = torch.cat((prev_actuation_sample_1, prev_actuation_sample_2, prev_actuation_sample_3), axis=0)
        return (
            message,
            room_name_sample,
            setpoint_sample,
            actuation_sample,
            prev_actuations_sample,
        )

if __name__ == '__main__':
    data = MessageDataset('../simulation_data.csv')

