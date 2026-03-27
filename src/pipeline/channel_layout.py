"""Shared EEG channel layout constants used across pipeline modules."""

EEG_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F9', 'F7', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F8', 'F10',
    'FT9', 'FT7',
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'FT8', 'FT10',
    'T9', 'T7',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'T8', 'T10',
    'TP9', 'TP7',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'TP10',
    'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'P10',
    'PO7', 'POz', 'PO8',
    'O1', 'O2',
]  # according to printout of quattrocento

EEG_CHANNELS_BY_AREA = {
    area_label: [
        ch for ch in EEG_CHANNELS
        if (ch[:len(area_abbr)] == area_abbr)
        and ((ch[len(area_abbr):].isnumeric()) or ch[len(area_abbr):] == 'z')
    ]
    for area_label, area_abbr in [
        ('Frontal Pole', 'Fp'), ('Anterior Frontal', 'AF'), ('Fronto-Central', 'FC'), ('Frontal', 'F'),
        ('Fronto-Temporal', 'FT'), ('Temporal', 'T'), ('Central', 'C'), ('Temporo-Parietal', 'TP'),
        ('Centro-Parietal', 'CP'), ('Parietal', 'P'), ('Parieto-Occipital', 'PO'), ('Occipital', 'O')
    ]
}
EEG_CHANNEL_IND_DICT = {ch: ind for ind, ch in enumerate(EEG_CHANNELS)}

