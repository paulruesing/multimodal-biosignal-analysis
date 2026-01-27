from pathlib import Path
from tqdm import tqdm
from src.pipeline.otb_file_handling import import_otb4_to_csv

if __name__ == '__main__':
    ROOT = Path().resolve().parent
    QTC_DATA = ROOT / 'data' / 'qtc_data'
    OTB_FILES = QTC_DATA / "otb-files"

    subject_ind = 5

    for i, title in tqdm(enumerate(
            [f"sub_{subject_ind:02}_eeg", f"sub_{subject_ind:02}_emg_1_flexor", f"sub_{subject_ind:02}_emg_2_extensor"]),
                         desc="Channel Subset Export"):
        try:
            result = import_otb4_to_csv(
                otb4_path=OTB_FILES / f"subject_{subject_ind:02}.otb4",
                output_dir=QTC_DATA / f"subject_{subject_ind:02}",
                output_title=title,
                channel_range=(64*i, 64*(i+1)),
                combine_channels=True,
                verbose=True
            )

            print("\n=== Import Summary ===")
            print(f"Device: {result['device']}")
            print(f"Total channels: {result['n_channels']}")
            print(f"Exported channels: {result['n_channels_exported']}")
            print(f"Channel range: {result['channel_range']}")
            print(f"Sampling freq: {result['sampling_freq']} Hz")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Data error: {e}")