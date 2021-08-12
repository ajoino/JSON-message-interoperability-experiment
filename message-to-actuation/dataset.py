from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List, Union

import pandas as pd
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
        self.simulation_data = pd.read_csv(file_path, sep=';').dropna().reset_index(drop=True)
        self.room_categories = pd.Categorical(self.simulation_data['room_name']).categories
        self.train = train
        self.validation = validation
        self.setpoint_transform = setpoint_transform
        self.actuation_transform = actuation_transform
        self.room_transform = room_transform

    def __len__(self) -> int:
        return 2*len(self.simulation_data)

    def __getitem__(self, idx: int) -> Tuple[List[Dict], int, float, float, float]: # Returns message, room label, setpoint, and actuation value
        message = self.simulation_data.loc[idx//2, 'messages_a'] if idx % 2 == 0 else self.simulation_data['messages_b'][idx//2]
        room_name_sample, setpoint_sample, actuation_sample, prev_actuation_sample = (
            self.simulation_data.loc[idx//2, column]
            for column in ('room_name', 'setpoint', 'actuation', 'previous_actuation')
        )

        if self.room_transform:
            room_name_sample = self.room_transform((self.room_categories.get_loc(room_name_sample), ))
        else:
            room_name_sample = self.room_categories(room_name_sample)
        if self.setpoint_transform:
            setpoint_sample = self.setpoint_transform((setpoint_sample, ))
        if self.actuation_transform:
            actuation_sample = self.actuation_transform((actuation_sample, ))
            prev_actuation_sample = self.actuation_transform((prev_actuation_sample, ))
        return (
            message,
            room_name_sample,
            setpoint_sample,
            actuation_sample,
            prev_actuation_sample
        )

if __name__ == '__main__':
    data = MessageDataset('../simulation_data.csv')

