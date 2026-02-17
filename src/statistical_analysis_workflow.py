from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import src.pipeline.signal_features as features
import src.pipeline.statistical_modelling as statistics
import src.pipeline.data_integration as data_integration
import src.pipeline.data_analysis as data_analysis
import src.pipeline.visualizations as visualizations
from src.pipeline.visualizations import EEG_CHANNEL_IND_DICT, EEG_CHANNELS_BY_AREA
import src.utils.file_management as filemgmt







if __name__ == '__main__':
    ######## PREPARATION #########
    ROOT = Path().resolve().parent
    DATA = ROOT / "data"
    OUTPUT = ROOT / 'output'
    STUDY_PLOTS = OUTPUT / 'plots' / 'data_analysis_plots'
    EXPERIMENT_DATA = DATA / "experiment_results"
    FEATURE_OUTPUT_DATA = DATA / "precomputed_features"
    STATISTICS_OUTPUT_DATA = OUTPUT / 'statistical_analysis'


    ##### PARAMETERS
    # feature extraction:
    n_within_trial_segments: int = 40 # slices per ~40s trial
    overwrite: bool = False  # ALWAYS compute new frame

    # data exploration:
    add_bin_features_dict: dict[str, int] = {'Median Force Level [0-1]': 4, 'Familiarity [0-7]': 5,
                                               'GSR [0-3.3]': 4, 'Trial ID': 4}
    # creates bin index for per subject values to be used as categories (new col. will be named "{OLD_COL}_bin")
    cmc_plot_categories: list[str] = ['Subject ID', 'Category or Silence', 'Familiarity [0-7]_bin', 'Trial ID_bin',
                                      'Median Force Level [0-1]_bin', 'GSR [0-3.3]_bin']
    # subject wise line plots:
    plot_cmc_lineplots: bool = False
    save_cmc_lineplots: bool = True
    # compound scatters:
    show_cmc_scatterplots: bool = False
    save_cmc_scatterplots: bool = True


    # statistical analysis:
    conduct_analysis: bool = True
    show_effect_plots: bool = True
    lvls_to_include: list[Literal['lvl0', 'lvl1', 'lvl2', 'lvl3', 'lvl4']] = ['lvl0', 'lvl1', 'lvl2', 'lvl3']



    #####################################################
    ################ FEATURE EXTRACTION #################
    #####################################################

    try:
        if overwrite: raise ValueError
        all_subject_data_frame = pd.read_csv(filemgmt.most_recent_file(FEATURE_OUTPUT_DATA, ".csv", [f"Combined Statistics {n_within_trial_segments}seg"]))  # pd.read_csv(FEATURE_OUTPUT_DATA / "statistics_temp_eeg_alpha.csv")  # set to None to run computation
        print("Fetching existing statistical data frame...\n")


    except ValueError:  # if no all subject dataframe found
        print("Computing new statistical data frame...\n")
        ### PSD PARAMETERS
        # average over below bands and channels (region_label labels the channel group):
        modality_region_channels_band_psd_list: list[tuple[str, str, list[str], str]] = [
            # MODALITY, REGION_LABEL, REGION_CHANNELS, BAND
            ('eeg', 'FC_CP_T',
             EEG_CHANNELS_BY_AREA['Fronto-Central'] + EEG_CHANNELS_BY_AREA['Centro-Parietal'] + EEG_CHANNELS_BY_AREA['Temporal'],
            'theta'),  # H2
            ('eeg', 'F_C',
             EEG_CHANNELS_BY_AREA['Frontal'] + EEG_CHANNELS_BY_AREA['Central'], 'beta'),  # H3
            ('eeg', 'P_PO',
             EEG_CHANNELS_BY_AREA['Parietal'] + EEG_CHANNELS_BY_AREA['Parieto-Occipital'], 'alpha'),  # H4
            ('eeg', 'Global', None, 'gamma'),
            ('emg_1_flexor', 'Global', None, 'all'),
            ('emg_2_extensor', 'Global', None, 'all')
        ]
        # select target band from freq_band_psd_per_segment_dict from
        #   - EMG   -> 'slow', 'fast'
        #   - EEG
        #       -> 'delta' (not sufficient frequencies)
        #       -> 'theta' (beat perception, entrainment)
        #       -> 'alpha' (auditory attention)
        #       -> 'beta' (motor control)
        #       -> 'gamma'

        # window lenghts:
        psd_time_window_size_sec = .25
        psd_is_log_scaled: bool = True  # define whether PSD was log scaled during feature extraction




        ### CMC PARAMETERS
        muscle_operator_band_cmc_list: list[tuple[str, str, str]] = [
            # MUSCLE, OPERATOR (max / mean / median), BAND
            ('Flexor', 'max', 'beta'),
            ('Flexor', 'max', 'gamma'),
            ('Flexor', 'mean', 'beta'),
            ('Flexor', 'mean', 'gamma'),
            ('Extensor', 'max', 'beta'),
            ('Extensor', 'max', 'gamma'),
            ('Extensor', 'mean', 'beta'),
            ('Extensor', 'mean', 'gamma'),
        ]
        # select target band from freq_band_psd_per_segment_dict from
        #   - EMG   -> 'slow', 'fast'
        #   - EEG
        #       -> 'delta' (not sufficient frequencies)
        #       -> 'theta' (beat perception, entrainment)
        #       -> 'alpha' (auditory attention)
        #       -> 'beta' (motor control)
        #       -> 'gamma'

        # window lengths:
        cmc_time_window_size_sec = 2.0





        ### DATA AGGREGATION PARAMETERS
        # how many segments:
        print(f"Will split 45sec trial into {n_within_trial_segments} segments (each ~{45/n_within_trial_segments:.1f}sec)")
        # standardization?
        modalities_to_standardize: list[str] = []  #['PSD', 'Force']
        music_features_to_fetch = ('BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean', 'IOI Variance Coeff',
                                   'Syncopation Ratio')







        ########### ITERATE OVER ALL PARTICIPANTS ###########
        all_subject_data_frame = pd.DataFrame(columns=['Subject ID'])
        for subject_ind in range(8):
            print("\n")
            print("-" * 100)
            print(f"------------     Aggregating data for subject\t\t{subject_ind:02}     ------------- ")
            print("-" * 100)

            # dependent directories:
            subject_psd_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
            subject_cmc_save_dir = FEATURE_OUTPUT_DATA / f"subject_{subject_ind:02}"
            subject_experiment_data_dir = EXPERIMENT_DATA / f"subject_{subject_ind:02}"








            ### IMPORT LOG AND SERIAL DATAFRAMES
            log_df = data_integration.fetch_enriched_log_frame(subject_experiment_data_dir)
            serial_df = data_integration.fetch_enriched_serial_frame(subject_experiment_data_dir)
            # make time-zone aware:
            log_df.index = data_analysis.make_timezone_aware(log_df.index)
            serial_df.index = data_analysis.make_timezone_aware(serial_df.index)

            # slice towards qtc measurements:
            qtc_start, qtc_end = data_integration.get_qtc_measurement_start_end(log_df, False)
            sliced_log_df = log_df[qtc_start:qtc_end]
            sliced_serial_df = serial_df[qtc_start:qtc_end]











            ### DERIVE SEGMENT TIMESPANS
            # trial start end times:
            # (contains default cut-off seconds to prevent transients!)
            trial_start_end_dict = data_integration.get_all_task_start_ends(log_df, 'dict')
            # convert into segment start end times:
            seg_starts = []; seg_ends = []
            for start, end in trial_start_end_dict.values():
                seg_starts_range = pd.date_range(start, end, periods=n_within_trial_segments+1, inclusive='both')
                for ind, seg_start in enumerate(seg_starts_range.values[:-1]):
                    seg_starts.append(seg_start)
                    seg_ends.append(seg_starts_range.values[ind+1])






            ### PREPARE DATAFRAME
            single_subject_data_frame = pd.DataFrame(index=range(len(seg_starts)))







            ### IMPORT AND AGGREGATE PSD DATA PER HYPOTHESIS (modality_region_channels_band_psd_list)
            # loop over configurations:
            for modality, region_label, channels, band in modality_region_channels_band_psd_list:
                # import PSD:
                psd_spectrograms, psd_times, psd_freqs = features.fetch_stored_spectrograms(
                    subject_psd_save_dir, modality='PSD', file_identifier=modality)
                #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                # convert PSD second time-centers into timestamps:
                psd_timestamps = data_analysis.add_time_index(
                    start_timestamp=qtc_start + pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                    end_timestamp=qtc_end - pd.Timedelta(seconds=psd_time_window_size_sec / 2),
                    n_timesteps=len(psd_times)
                )
                psd_timestamps = data_analysis.make_timezone_aware(psd_timestamps)

                # takes shapes (n_windows, n_freqs, n_channels)
                psd_aggregated = features.aggregate_psd_spectrogram(psd_spectrograms, psd_freqs, normalize_mvc=False,
                                                           channel_indices=[EEG_CHANNEL_IND_DICT[ch] for ch in channels] if channels is not None else None,
                                                           is_log_scaled=psd_is_log_scaled, freq_slice=band,
                                                           aggregation_ops=[('mean', 1),   # mean within freq band
                                                                            # mean over EEG channels, max over EMG ones:
                                                                            ('mean' if 'eeg' in modality else 'max', 1),
                                                                            ],)
                # returns shape (n_windows, )

                # split per segment:
                psd_per_segment = data_analysis.apply_window_operator(
                    window_timestamps=seg_starts,
                    window_timestamps_ends=seg_ends,

                    target_array=psd_aggregated,
                    target_timestamps=psd_timestamps,

                    operation='mean',
                    axis=0,  # time axis
                )  # (n_trials, )

                # save to dataframe:
                single_subject_data_frame[f"PSD_{modality}_{region_label}_{band}"] = psd_per_segment








                ### IMPORT AND AGGREGATE CMC DATA PER HYPOTHESIS (muscle_operator_band_cmc_list)
                # loop over configurations:
                for muscle, operator, band in muscle_operator_band_cmc_list:
                    # import CMC:
                    cmc_spectrograms, cmc_times, cmc_freqs = features.fetch_stored_spectrograms(
                        subject_cmc_save_dir, modality='CMC', file_identifier=muscle)
                    #   -> shape: (n_windows, n_freqs, n_channels), (n_windows), (n_freqs)
                    # convert CMC second time-centers into timestamps:
                    cmc_timestamps = data_analysis.add_time_index(
                        start_timestamp=qtc_start + pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        end_timestamp=qtc_end - pd.Timedelta(seconds=cmc_time_window_size_sec / 2),
                        n_timesteps=len(cmc_times)
                    )
                    cmc_timestamps = data_analysis.make_timezone_aware(cmc_timestamps)

                    # takes shapes (n_windows, n_freqs, n_channels)
                    cmc_aggregated = features.aggregate_psd_spectrogram(cmc_spectrograms, cmc_freqs, normalize_mvc=False,
                                                               is_log_scaled=False, freq_slice=band,
                                                               aggregation_ops=[('max', 1),  # mean within freq band
                                                                                # either take peak or average over channels
                                                                                (operator, 1),
                                                                                ]
                                                               )
                    # returns shape (n_windows, )

                    # split per segment:
                    cmc_per_segment = data_analysis.apply_window_operator(
                        window_timestamps=seg_starts,
                        window_timestamps_ends=seg_ends,

                        target_array=cmc_aggregated,
                        target_timestamps=cmc_timestamps,

                        operation='mean',
                        axis=0,  # time axis
                    )  # (n_trials, )

                    # save to dataframe:
                    single_subject_data_frame[f"CMC_{muscle}_{operator}_{band}"] = cmc_per_segment











            ### INDEPENDENT VARIABLE AGGREGATION
            # force level:
            force_level_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['Task-wise Scaled Force'],
                operation='median',
                axis=0,  # time axis
            )

            # trial category:
            song_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Song ID'],
                operation='mode',  # most common string value
                axis=0,  # time axis
            )
            silence_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Silence ID'],
                operation='mode',  # most common string value
                axis=0,  # time axis
            )
            is_music_trial = [not pd.isna(song_id) and pd.isna(silence_id) for song_id, silence_id in zip(song_id_per_segment, silence_id_per_segment)]

            # trial ID:
            trial_id_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Trial ID'],
                operation='mode', axis=0,
            )

            # musical features:
            music_feature_tuples = [
                data_integration.fetch_music_features(log_df, trial_id=trial_id,
                                                      features_to_return=music_features_to_fetch) for trial_id in trial_id_per_segment
            ]


            # song category:
            song_category_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Perceived Category'],
                operation='mode', axis=0,
            )
            # song category with missing values replaced with: 'Silence'
            category_or_silence = pd.Series(song_category_per_segment).fillna('Silence')

            song_familiarity_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Familiarity'],
                operation='mode', axis=0,
            )
            emotional_state_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Emotional State'],
                operation='mode', axis=0,
            )

            song_liking_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Liking'],
                operation='mode', axis=0,
            )

            task_frequency_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=log_df['Task Frequency'],
                operation='mode', axis=0,
            )

            # heart rate::
            bpm_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['bpm'],
                operation='median',
                axis=0,  # time axis
            )

            # heart rate variability:
            hrv_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['hrv'],
                operation='median',
                axis=0,  # time axis
            )

            # galvanic skin response:
            gsr_per_segment = data_analysis.apply_window_operator(
                window_timestamps=seg_starts,
                window_timestamps_ends=seg_ends,
                target_array=sliced_serial_df['gsr'],
                operation='median',
                axis=0,
            )



            ### APPEND TO PER SUBJECT DATAFRAME
            # append to subject df:
            for column_name, data in [('Subject ID', [subject_ind] * len(seg_starts)),
                                      ('Trial ID', trial_id_per_segment),
                                      ('Music Listening', is_music_trial),
                                      ('Median Force Level [0-1]', force_level_per_segment),
                                      ('Task Frequency', task_frequency_per_segment),
                                      ('Emotional State [0-7]', emotional_state_per_segment),
                                      ('Median Heart Rate [bpm]', bpm_per_segment),
                                      ('Median HRV [sec]', hrv_per_segment),
                                      ('GSR [0-3.3]', gsr_per_segment),
                                      # music features:
                                      ('Perceived Category', song_category_per_segment),
                                      ('Category or Silence', category_or_silence),
                                      ('Liking [0-7]', song_liking_per_segment),
                                      ('Familiarity [0-7]', song_familiarity_per_segment),
                                      (list(music_features_to_fetch), music_feature_tuples),
                                      ]:
                single_subject_data_frame[column_name] = data

            # standardize:
            for modality in modalities_to_standardize:
                for column in [c for c in single_subject_data_frame.columns if modality in c]:
                    # columns to standardize:
                    print("Standardizing statistics for: ", column)
                    single_subject_data_frame[column] = single_subject_data_frame[column].transform(lambda x: (x - x.mean()) / x.std())






            ### CONCAT WITH ALL SUBJECT DATAFRAME
            all_subject_data_frame = pd.concat([all_subject_data_frame, single_subject_data_frame], axis=0)






        ######### SAVE COMBINED STATISTICS #########
        all_subject_data_frame.to_csv(FEATURE_OUTPUT_DATA / filemgmt.file_title(f"Combined Statistics {int(n_within_trial_segments)}seg", ".csv"), index=False)





    #######################################################
    ################ DATA EXPLORATION #####################
    #######################################################

    all_subject_data_frame = data_analysis.create_trial_bins(
        df=all_subject_data_frame,
        columns_to_bin=list(add_bin_features_dict.keys()),
        n_bins_dict=add_bin_features_dict,

    )



    if plot_cmc_lineplots:
        cmc_operator: Literal['mean', 'max'] = 'mean'
        print(features.compute_cmc_independence_threshold(5, .2))
        cmc_plot_min, cmc_plot_max = .7, 1.0  # independence thresholds: 80% -> .673, 90% -> .753
        n_yticks: int = 4
        include_std_dev: bool = True
        std_dev_factor: float = .2
        colormap = 'tab20'

        # plot CMC per subject and category:
        for muscle in ['Flexor', 'Extensor']:

            # loop over categories (new plot per category)
            for category_column in cmc_plot_categories:
                if category_column == 'Subject ID': continue  # doesn't work, we split by subject anyways
                print(f"Plotting CMC lineplot for category: {category_column}")
                unique_labels = all_subject_data_frame[category_column].dropna().unique().tolist()
                unique_labels.sort()  # sort values
    
                # sample colors evenly from colormap
                cmap = plt.colormaps[colormap]  # or 'viridis', 'plasma', 'Set3', etc.
                colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

                # create legend handles for all unique labels (to ensure complete legend):
                from matplotlib.lines import Line2D
                legend_handles = [Line2D([0], [0], color=color, lw=2, label=label)
                                  for color, label in zip(colors, unique_labels)]
    
                # subplot per subject and band:
                subject_ids = all_subject_data_frame['Subject ID'].unique().tolist()
                fig, axs = plt.subplots(2, # top: beta, bottom: gamma
                                        len(subject_ids), figsize=(20, 10))
    
                # loop over bands:
                for row_ind, freq_band in enumerate(['beta', 'gamma']):
    
                    # loop over subjects:
                    for col_ind, subject_id in enumerate(subject_ids):
                        # derive relevant data:
                        subject_frame = all_subject_data_frame[all_subject_data_frame['Subject ID'] == subject_id]
    
                        # prepare x-labels: (0 = trial start, 1 = trial end)
                        x_ticks = np.linspace(0, 1,
                                              max(n_within_trial_segments, 2))  # at least 2 (should plot line not point)
    
                        for color, category in zip(colors, unique_labels):
                            category_frame = subject_frame[subject_frame[category_column] == category]
    
                            # averaging is important if there are multiple trials per category:
                            # (and does no harm otherwise)
                            within_trial_counter = category_frame.groupby('Trial ID').cumcount()
                            grouped_cmc_timesteps = category_frame[f"CMC_{muscle}_{cmc_operator}_{freq_band}"].groupby(within_trial_counter)
                            cmc_series = grouped_cmc_timesteps.mean().to_numpy()
                            cmc_std = grouped_cmc_timesteps.std().to_numpy()
    
                            if len(cmc_series) == 0: continue  # no entries for that category instance
    
                            if len(cmc_series) == 1:  # if only one sample, draw line between start and end
                                cmc_series = np.array([cmc_series[0], cmc_series[0]])
                                cmc_std = np.array([cmc_std[0], cmc_std[0]])
    
                            # plot line with std. dev:
                            axs[row_ind, col_ind].plot(x_ticks, cmc_series, label=category, color=color)
                            if include_std_dev:
                                axs[row_ind, col_ind].fill_between(x_ticks, cmc_series - std_dev_factor * cmc_std, cmc_series + std_dev_factor * cmc_std, alpha=0.2,
                                                                   color=color)
    
                        # formatting:
                        if row_ind == 0:  # titles for top row
                            axs[row_ind, col_ind].set_title(f"Subject {subject_id:02}")
                        # y label for left column:
                        if col_ind == 0:
                            axs[row_ind, col_ind].set_ylabel(f"{muscle} {freq_band.capitalize()} CMC (Mean{f' ± {std_dev_factor:.1f}x Std.Dev.' if include_std_dev else ''})")
                        else:
                            axs[row_ind, col_ind].set_ylabel('')
                        # yticks:
                        y_ticks = np.linspace(cmc_plot_min, cmc_plot_max, n_yticks)
                        axs[row_ind, col_ind].set_yticks(y_ticks)  # no effect, but prevents warning for below
                        if col_ind != 0:  # remove y labels:
                            axs[row_ind, col_ind].set_yticklabels([''] * len(y_ticks))
                        # equal y-axis limits:
                        axs[row_ind, col_ind].set_ylim([cmc_plot_min, cmc_plot_max])
                        # grid:
                        axs[row_ind, col_ind].grid()
                        # legend:
                        if row_ind == 1 and col_ind == (len(subject_ids) - 1):  # lower right plot receives legend
                            axs[row_ind, col_ind].legend(handles=legend_handles, bbox_to_anchor=(1.0, 0.0),
                                                         loc='lower left', title=category_column,)
    
                        # x ticks and label:
                        if row_ind == 1:  # only lower row
                            if len(x_ticks) > 2:  # insert middle
                                x_tick_labels = ['Start'] + [''] * (len(x_ticks) - 2) + ['End']
                            else:
                                x_tick_labels = ['Start', 'End']
                            # x label:
                            axs[row_ind, col_ind].set_xlabel("Trial Time")
                        else:
                            x_tick_labels = [""] * len(x_ticks)
                            axs[row_ind, col_ind].set_xlabel('')
                        # enforce x ticks:
                        axs[row_ind, col_ind].set_xticks(x_ticks)
                        axs[row_ind, col_ind].set_xticklabels(x_tick_labels)

                # figure title:
                fig.suptitle(f"CMC per Subject and {category_column}")
    
                # (eventually) save figure:
                if save_cmc_lineplots:
                    save_path = STATISTICS_OUTPUT_DATA / filemgmt.file_title(f"CMC {muscle} per Subject per {category_column}", ".svg")
                    fig.savefig(save_path, bbox_inches='tight')
                
                # show:
                plt.show()





    # dependent var (plot beta + gamma scatter for flexor + extensor) for each category:
    # [((x_column, x_label), (y_column, y_label), category_column), ...]
    if show_cmc_scatterplots:
        scatters_to_plot: list[tuple[tuple[str, str], tuple[str, str], str]] = []
        for category_column in cmc_plot_categories:
            scatters_to_plot.append(
                (('CMC_Flexor_mean_beta', 'CMC Flexor Avg. Beta'), ('CMC_Flexor_mean_gamma', 'CMC Flexor Avg. Gamma'),
                 category_column))
            scatters_to_plot.append(
                (('CMC_Extensor_mean_beta', 'CMC Extensor Avg. Beta'), ('CMC_Extensor_mean_gamma', 'CMC Extensor Avg. Gamma'),
                 category_column))


        # scatters:
        for (x, x_label), (y, y_label), category_column in scatters_to_plot:
            dataframe_subset = all_subject_data_frame.dropna(subset=[x, y, category_column])
            x_data = dataframe_subset[x]
            y_data = dataframe_subset[y]
            category_list = dataframe_subset[category_column]
            visualizations.plot_scatter(x=x_data, x_label=x_label,
                                        y=y_data, y_label=y_label,
                                        category_list=category_list,
                                        category_label=category_column,
                                        save_dir=STATISTICS_OUTPUT_DATA,
                                        cmap='tab20',
                                        )




    #######################################################
    ################ STATISTICAL ANALYSIS #################
    #######################################################

    if conduct_analysis:
        # Store all results and diagnostics for summary tables
        all_model_results = []
        all_diagnostics = []

        for hypothesis, dependent_variable in [
            # CMC Hypotheses:
            ('H1: Flexor Beta Peak CMC Increases with Music', "CMC_Flexor_max_beta"),
            ('H1: Flexor Beta Avg. CMC Increases with Music', "CMC_Flexor_mean_beta"),
            ('H1: Flexor Gamma Peak CMC Increases with Music', "CMC_Flexor_max_gamma"),
            ('H1: Flexor Gamma Avg. CMC Increases with Music', "CMC_Flexor_mean_gamma"),
            ('H1: Extensor Beta Peak CMC Increases with Music', "CMC_Extensor_max_beta"),
            ('H1: Extensor Beta Avg. CMC Increases with Music', "CMC_Extensor_mean_beta"),
            ('H1: Extensor Gamma Peak CMC Increases with Music', "CMC_Extensor_max_gamma"),
            ('H1: Extensor Gamma Avg. CMC Increases with Music', "CMC_Extensor_mean_gamma"),

            # EEG-PSD Hypotheses:
            ('H2: Temporal Prediction PSD Increases with Music', 'PSD_eeg_FC_CP_T_theta'),
            ('H3: Vigilance PSD Increases with Music', 'PSD_eeg_F_C_beta'),
            ('H4: Internal vs. External Attention PSD changes with Music', 'PSD_eeg_P_PO_alpha'),
            ('H5: Long Range Interactions PSD Increases with Music', 'PSD_eeg_Global_gamma'),

            # Validation Hypotheses (EMG PSD):
            ('VALIDATION: EMG Flexor PSD Increases with Force', 'PSD_emg_1_flexor_Global_all'),
            ('VALIDATION: EMG Extensor PSD Increases with Force', 'PSD_emg_2_extensor_Global_all'), ]:

            # Intro String:
            print("\n")
            print("=" * 100)
            print("=" * 100)
            print(f"HYPOTHESIS:\t\t{hypothesis} ")
            print(f"DEPENDENT VARIABLE:\t{dependent_variable}")
            print("=" * 100)
            print("=" * 100, "\n"*3)


            ############# COMPARISON LEVELS #############
            if "lvl0" in lvls_to_include:
                # Level 0: Music + Force + Trial ID (all data)
                level0_results = statistics.fit_both_models(
                    df=all_subject_data_frame,
                    response_var=dependent_variable,
                    condition_vars={'Music Listening': 'categorical'},
                    reference_categories={'Music Listening': 'False'},
                    explanatory_vars=['Median Force Level [0-1]', 'Trial ID'],
                    comparison_level_name='Level 0 (Music + Force + Trial ID)',
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments
                )

                statistics.store_model_results(
                    model_results=level0_results,
                    hypothesis_name=hypothesis,
                    dependent_variable=dependent_variable,
                    comparison_level_name='Level 0 (Music + Force + Trial ID)',
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics
                )

            if "lvl1" in lvls_to_include:
                # Level 1: Music Categories or Silence
                level1_results = statistics.fit_both_models(
                    df=all_subject_data_frame,
                    response_var=dependent_variable,
                    condition_vars={
                        # changed:
                        'Category or Silence': 'categorical',
                    },
                    reference_categories={'Category or Silence': 'Silence'},
                    explanatory_vars=['Median Force Level [0-1]', 'Trial ID'],
                    comparison_level_name='Level 1 (Category or Silence + Force + Trial ID)',
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments
                )

                statistics.store_model_results(
                    model_results=level1_results,
                    hypothesis_name=hypothesis,
                    dependent_variable=dependent_variable,
                    comparison_level_name='Level 1 (Category or Silence + Force + Trial ID)',
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics
                )

            if "lvl2" in lvls_to_include:
                # Level 2: Subjective Music Features (music trials only)
                level2_results = statistics.fit_both_models(
                    df=all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
                    response_var=dependent_variable,
                    condition_vars={
                        # changed:
                        'Perceived Category': 'categorical',
                        'Familiarity [0-7]': 'ordinal'
                    },
                    reference_categories={'Perceived Category': 'Classical'},
                    explanatory_vars=['Median Force Level [0-1]', 'Trial ID',
                                      # new:
                                      'Liking [0-7]'],
                    comparison_level_name='Level 2 (Subjective Music Features + Force + Trial ID)',
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments
                )

                statistics.store_model_results(
                    model_results=level2_results,
                    hypothesis_name=hypothesis,
                    dependent_variable=dependent_variable,
                    comparison_level_name='Level 2 (Subjective Music Features + Force + Trial ID)',
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics
                )

            if "lvl3" in lvls_to_include:
                # Level 3: Add Emotional State and Biomarkers (music trials only)
                level3_results = statistics.fit_both_models(
                    df=all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
                    response_var=dependent_variable,
                    condition_vars={
                        'Perceived Category': 'categorical',
                        'Familiarity [0-7]': 'ordinal'
                    },
                    reference_categories={'Perceived Category': 'Classical'},
                    explanatory_vars=['Median Force Level [0-1]', 'Trial ID', 'Liking [0-7]',
                                      # new:
                                      'Emotional State [0-7]',
                                      'Median Heart Rate [bpm]', 'Median HRV [sec]', 'GSR [0-3.3]'],
                    comparison_level_name='Level 3 (Emotional State + Biomarkers + Subjective Music Features + Force + Trial ID)',
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments
                )

                statistics.store_model_results(
                    model_results=level3_results,
                    hypothesis_name=hypothesis,
                    dependent_variable=dependent_variable,
                    comparison_level_name='Level 3 (Emotional State + Biomarkers + Subjective Music Features + Force + Trial ID)',
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics
                )

            if "lvl4" in lvls_to_include:
                # Level 4: Add Objective Music Features (music trials only)
                level4_results = statistics.fit_both_models(
                    df=all_subject_data_frame.loc[all_subject_data_frame['Music Listening']],
                    response_var=dependent_variable,
                    condition_vars={
                        'Perceived Category': 'categorical',
                        'Familiarity [0-7]': 'ordinal'
                    },
                    reference_categories={'Perceived Category': 'Classical'},
                    explanatory_vars=['Median Force Level [0-1]', 'Trial ID', 'Liking [0-7]',
                                      'Emotional State [0-7]',
                                      'Median Heart Rate [bpm]', 'Median HRV [sec]', 'GSR [0-3.3]',
                                      # new:
                                      'BPM_manual', 'Spectral Flux Mean', 'Spectral Centroid Mean', 'IOI Variance Coeff',
                                      'Syncopation Ratio'],
                    comparison_level_name='Level 4 (Objective Music Features + Emotional State + Biomarkers + Subjective Music Features + Force + Trial ID)',
                    hypothesis_name=hypothesis,
                    n_windows_per_trial=n_within_trial_segments
                )

                statistics.store_model_results(
                    model_results=level4_results,
                    hypothesis_name=hypothesis,
                    dependent_variable=dependent_variable,
                    comparison_level_name='Level 4 (Objective Music Features + Emotional State + Biomarkers + Subjective Music Features + Force + Trial ID)',
                    all_results_list=all_model_results,
                    diagnostics_list=all_diagnostics
                )


        # ============================================================================
        # Summary statistics
        # ============================================================================

        # Generate all summary tables with one function call
        statistics.generate_all_summary_tables(
            results_df=pd.DataFrame(all_model_results),
            output_dir=STATISTICS_OUTPUT_DATA,
            diagnostics_df=pd.DataFrame(all_diagnostics),
            file_identifier=f"{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
        )



        # ============================================================================
        # Subject-specific analysis
        # ============================================================================

        # After all models are run
        filemgmt.assert_dir(STATISTICS_OUTPUT_DATA / f"subject_level_effects_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}")
        statistics.create_subject_effect_summary(
            all_model_results=all_model_results,
            original_data=all_subject_data_frame,
            output_dir=STATISTICS_OUTPUT_DATA / f"subject_level_effects_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}"
        )

        # ============================================================================
        # Plotting
        # ============================================================================

        results_frame = pd.DataFrame(all_model_results)

        # derive unique hypotheses:
        hypotheses = results_frame['Hypothesis'].dropna().unique().tolist()

        # group hypotheses:
        cmc_flexor_hypotheses = [h for h in hypotheses if 'CMC' in h and 'Flexor' in h and not 'VALIDATION: ' in h]
        cmc_extensor_hypotheses = [h for h in hypotheses if 'CMC' in h and 'Extensor' in h and not 'VALIDATION: ' in h]
        psd_hypotheses = [h for h in hypotheses if 'PSD' in h and not 'VALIDATION: ' in h]
        validation_hypotheses = [h for h in hypotheses if 'VALIDATION: ' in h]



        visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_flexor_hypotheses, output_dir=STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H1_Flexor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='OLS', hidden=not show_effect_plots)
        visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_flexor_hypotheses, output_dir=STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H1_Flexor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='LME', hidden=not show_effect_plots)

        visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='OLS', hidden=not show_effect_plots)
        visualizations.plot_hypothesis_forest_mosaic(results_frame, cmc_extensor_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H1_Extensor_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='LME', hidden=not show_effect_plots)

        visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='OLS', hidden=not show_effect_plots)
        visualizations.plot_hypothesis_forest_mosaic(results_frame, psd_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"H2-5_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='LME', hidden=not show_effect_plots)

        visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='OLS', hidden=not show_effect_plots)
        visualizations.plot_hypothesis_forest_mosaic(results_frame, validation_hypotheses, output_dir = STATISTICS_OUTPUT_DATA,
                                      file_identifier_suffix=f"VAL_{n_within_trial_segments}seg_{"_".join(lvls_to_include)}",
                                      model_type='LME', hidden=not show_effect_plots)