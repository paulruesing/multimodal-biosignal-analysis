from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import umap
import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

import src.utils.file_management as filemgmt
from src.pipeline.data_integration import fetch_enriched_log_frame, get_all_task_start_ends
from src.pipeline.music_control import SpotifyController, compute_all_musical_features, add_metrics_from_txt
import src.pipeline.visualizations as visualizations


mpl.use('Qt5Agg')


if __name__ == '__main__':
    ######################### PARAMETERS #########################
    ROOT = Path().resolve().parent
    AUDIO_DIR = ROOT / 'data' / 'songs'
    RESULT_DIR = ROOT / 'data' / 'song_characteristics'
    filemgmt.assert_dir(RESULT_DIR)
    AUDIO_CONFIG = ROOT / "config" / "music_selection.txt"
    PLOT_DIR = ROOT / "output" / "music_characteristics"  # set to None if no plot saving desired
    filemgmt.assert_dir(PLOT_DIR)

    ##### LOOKUP TABLE COMPUTATION #####
    ### Step 1:
    recompute_feature_lookup_table: bool = False  # compute music metrics (BPM, Syncopation, Spectral Flux, ...)
    ### Step 2:
    extend_lookup_table_from_config: bool = False  # extend computed metrics by Song Title, Artist & Manual BPM
    ### Step 3:
    # single songs to add to existing look up table:
    # below structure [(filepath, Artist, Title, Category, Genre, Spotify URL, Start After), ...]:
    single_files_to_add: list[tuple[str, str, str, str, str, str, float]] = [
        # these two were accidentally played but should be included in the analysis:
        # ("Soulsearcher - Can't Get Enough (Jazz N Groove Nu Disco Vocal) - Defected Records.mp3", "Soulsearcher", "Can't Get Enough (Jazz N Groove Nu Disco Vocal)", "Unfamiliar Groovy", "Disco House", "https://open.spotify.com/intl-de/track/13qX3v31O0UBg59v3BC6fU?si=e13034bae8fa4959", 0),
    ]


    ##### DESCRIPTIVE STATISTICS #####
    N_SUBJECTS: int = 12

    analyse_category_fit: bool = True  # plot, how many participants RE-assigned categories
    cluster_results: bool = False
    compute_mutual_information: bool = False  # analyse relation music features -> genres / categories
    plot_scatters: bool = True  # plot feature distribution across categories

    # plot parameters:
    song_colors = {'Classic': 'orange', 'Groovy': 'red',
                   'Happy': 'green', 'Sad': 'blue'}

    # for MI and Scatter hue:
    target_label: Literal['Genre', 'Category'] = 'Category'

    ### scatter parameters:
    # x: spectral flux std, y: spectral flux mean
    scatter_x_y_combinations = [
        ('Spectral Centroid Mean', ('Spectral Flux Std.', 'Spectral Flux SD')),
        (('IOI Variance Coeff', 'IOI Variance Coefficient'), ('BPM_manual', 'BPM')),
    ]
    # -> MI Candidates:
    #       'Spectral Centroid Mean', 'Spectral Flux Std.', 'IOI Variance Coeff',
    #       'Spectral Flux Mean', 'Spectral Flux Min.', 'BPM_manual'
    # -> which of these linear?
    #       'Spectral Centroid Mean' for all but happy
    #       'Spectral Flux Std' for happy vs. sad
    #       'IOI Variance Coeff' for all but happy




    ######################### WORKFLOW #########################
    if recompute_feature_lookup_table:
        # initialise spotify controller:
        controller = SpotifyController(AUDIO_CONFIG)
        # we'll iterate over controller.category_url_dict to compute the characteristics

        # feature lists:
        categories = []
        genres = []
        spotify_urls = []
        start_afters = []
        file_names = []
        bpm_list = []
        spectral_flux_mins = []
        spectral_flux_maxs = []
        spectral_flux_means = []
        spectral_flux_stds = []
        spectral_centroid_mins = []
        spectral_centroid_maxs = []
        spectral_centroid_means = []
        ioi_var_coeffs = []
        syncopation_degrees = []
        syncopation_ratios = []

        # iterate over all defined songs and compute characteristics:
        for category, song_list in controller.category_url_dict.items():
            for genre, spotify_url, start_after, bpm, audio_file_title in tqdm(song_list, desc=f'Computing Metrics for Category {category}'):
                # sanity checks
                if audio_file_title.startswith('.'): continue
                audio_path = AUDIO_DIR / audio_file_title

                ### compute musical features:
                bpm_from_beats, spectral_flux_normalized, spectral_centroid, ioi_cv, syncopation_degree, syncopation_ratio = compute_all_musical_features(
                    audio_path, verbose=False)

                ### save data:
                # controller data:
                categories.append(category)
                genres.append(genre)
                spotify_urls.append(spotify_url)
                start_afters.append(start_after)
                if abs(bpm_from_beats - bpm) > .5: print(
                    f"Mismatch between calculated bpm ({bpm_from_beats:.2f}) and defined bpm ({bpm:.2f}) larger than .5! (song: {audio_file_title})")

                # computed data:
                file_names.append(audio_file_title)
                bpm_list.append(bpm_from_beats)
                spectral_flux_mins.append(spectral_flux_normalized.min())
                spectral_flux_maxs.append(spectral_flux_normalized.max())
                spectral_flux_means.append(spectral_flux_normalized.mean())
                spectral_flux_stds.append(spectral_flux_normalized.std())
                spectral_centroid_mins.append(spectral_centroid.min())
                spectral_centroid_maxs.append(spectral_centroid.max())
                spectral_centroid_means.append(spectral_centroid.mean())
                ioi_var_coeffs.append(ioi_cv)
                syncopation_degrees.append(syncopation_degree)  # (0 = strictly on-beat, 1 = highly syncopated)
                syncopation_ratios.append(syncopation_ratio)  # syncopated onsets (>0.2 beat away)


        # save as dataframe and export:
        frame = pd.DataFrame(index=file_names, data={
            'Category': categories,
            'Genre': genres,
            'Spotify URL': spotify_urls,
            'Intended Start [sec]': start_afters,
            'BPM': bpm_list,
            'Spectral Flux Min.': spectral_flux_mins,
            'Spectral Flux Max.': spectral_flux_maxs,
            'Spectral Flux Mean': spectral_flux_means,
            'Spectral Flux Std.': spectral_flux_stds,
            'Spectral Centroid Min': spectral_centroid_mins,
            'Spectral Centroid Max': spectral_centroid_maxs,
            'Spectral Centroid Mean': spectral_centroid_means,
            'IOI Variance Coeff': ioi_var_coeffs,
            'Syncopation Degree': syncopation_degrees,
            'Syncopation Ratio': syncopation_ratios,
        })
        print(frame)
        frame.to_csv(RESULT_DIR / filemgmt.file_title("Song Characteristic Lookup Table", ".csv"))

    else:  # load last results
        frame = pd.read_csv(filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Song Characteristic Lookup Table"]))
        print(f"Imported music characteristics for {len(frame)} songs")

        # select numerical and remove start parameter:
        feature_labels = [col for col in frame.columns if frame.dtypes[col] != 'object' and col != 'Intended Start [sec]']
        feature_array = frame.loc[:, feature_labels].to_numpy()

        #
        print("Feature indices: ", list(enumerate(feature_labels)))
        categories = frame['Category'].to_list()
        genres = frame['Genre'].to_list()





    ### EXTEND METRICS
    if extend_lookup_table_from_config:
        previous_frame = pd.read_csv(filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Song Characteristic Lookup Table"]))
        previous_frame.set_index("Unnamed: 0", inplace=True)
        print(previous_frame)
        new_frame = add_metrics_from_txt(previous_frame, AUDIO_CONFIG)
        new_frame.to_csv(RESULT_DIR / filemgmt.file_title("Extended Song Characteristic Lookup Table", ".csv"))

        ### ADD NEW SINGLE ENTRIES
        if len(single_files_to_add) > 0:
            current_frame = pd.read_csv(
                filemgmt.most_recent_file(RESULT_DIR, ".csv", ["Extended Song Characteristic Lookup Table"]))
            current_frame.drop(columns=[col for col in current_frame.columns if 'Unnamed' in col], inplace=True)

            # feature lists:
            artists = []
            titles = []
            manual_bpms = []
            categories = []
            genres = []
            spotify_urls = []
            start_afters = []
            file_names = []
            bpm_list = []
            spectral_flux_mins = []
            spectral_flux_maxs = []
            spectral_flux_means = []
            spectral_flux_stds = []
            spectral_centroid_mins = []
            spectral_centroid_maxs = []
            spectral_centroid_means = []
            ioi_var_coeffs = []
            syncopation_degrees = []
            syncopation_ratios = []

            for audio_file_title, artist, title, category, genre, spotify_url, start_after in single_files_to_add:
                bpm_from_beats, spectral_flux_normalized, spectral_centroid, ioi_cv, syncopation_degree, syncopation_ratio = compute_all_musical_features(
                    AUDIO_DIR / audio_file_title, verbose=False)

                artists.append(artist)
                titles.append(title)
                categories.append(category)
                genres.append(genre)
                spotify_urls.append(spotify_url)
                start_afters.append(start_after)

                # computed data:
                file_names.append(audio_file_title)
                bpm_list.append(bpm_from_beats)
                manual_bpms.append(bpm_from_beats)
                spectral_flux_mins.append(spectral_flux_normalized.min())
                spectral_flux_maxs.append(spectral_flux_normalized.max())
                spectral_flux_means.append(spectral_flux_normalized.mean())
                spectral_flux_stds.append(spectral_flux_normalized.std())
                spectral_centroid_mins.append(spectral_centroid.min())
                spectral_centroid_maxs.append(spectral_centroid.max())
                spectral_centroid_means.append(spectral_centroid.mean())
                ioi_var_coeffs.append(ioi_cv)
                syncopation_degrees.append(syncopation_degree)  # (0 = strictly on-beat, 1 = highly syncopated)
                syncopation_ratios.append(syncopation_ratio)  # syncopated onsets (>0.2 beat away)

            # create new rows:
            new_rows = pd.DataFrame(index=file_names, data={
                'Category': categories,
                'Genre': genres,
                'Spotify URL': spotify_urls,
                'Intended Start [sec]': start_afters,
                'BPM': bpm_list,
                'Spectral Flux Min.': spectral_flux_mins,
                'Spectral Flux Max.': spectral_flux_maxs,
                'Spectral Flux Mean': spectral_flux_means,
                'Spectral Flux Std.': spectral_flux_stds,
                'Spectral Centroid Min': spectral_centroid_mins,
                'Spectral Centroid Max': spectral_centroid_maxs,
                'Spectral Centroid Mean': spectral_centroid_means,
                'IOI Variance Coeff': ioi_var_coeffs,
                'Syncopation Degree': syncopation_degrees,
                'Syncopation Ratio': syncopation_ratios,
                'Title': titles,
                'Artist': artists,
                'BPM_manual': manual_bpms,
            })

            # and append
            new_frame = pd.concat([current_frame, new_rows], ignore_index=True)

            new_frame.to_csv(RESULT_DIR / filemgmt.file_title("Extended Song Characteristic Lookup Table", ".csv"),
                             index=False)



    if analyse_category_fit:
        from collections import defaultdict, Counter
        # subject_id -> categories
        category_reassigment_frame = pd.DataFrame()

        for subject_id in tqdm(range(N_SUBJECTS), desc="Fetching Categories per Trial and Subject"):
            subject_str = f"subject_{subject_id:02}"
            subject_dir = ROOT / "data" / "experiment_results" / subject_str

            # import enriched log_frames (with music category and perceived category):
            subject_log_frame = fetch_enriched_log_frame(subject_dir, verbose=False)

            # frame holds 'Music Category' and 'Perceived Category':
            # remove Familiar / Unfamiliar from Music Category
            subject_log_frame['Music Category'] = (
                subject_log_frame['Music Category']
                .str.replace(r"^(?:Unfamiliar|Familiar)\s+(?:\.{3,}\s*)?", "", regex=True)
                .str.strip()
            )

            # read value per task:
            original_categories = []
            perceived_categories = []
            song_titles = []
            song_artists = []
            subject_task_start_ends = get_all_task_start_ends(subject_log_frame, 'list')
            for start, _ in subject_task_start_ends:
                subset = subject_log_frame.loc[start:].iloc[0]
                if np.isnan(subset['Song ID']): continue   # skip silence trials
                original_categories.append(subset['Music Category'])
                perceived_categories.append(subset['Perceived Category'])
                song_titles.append(subset['Song Title'])
                song_artists.append(subset['Song Artist'])


            # result frame with re-assignments:
            per_subject_reassignment_frame = pd.DataFrame(
                data={'from': original_categories, 'to': perceived_categories, 'Title': song_titles, 'Artist': song_artists},
            )
            per_subject_reassignment_frame['Subject ID'] = subject_id

            # append to complete results:
            category_reassigment_frame = pd.concat([category_reassigment_frame, per_subject_reassignment_frame], axis=0)


        # overwrite the existing category target frame by max-pooling:
        common_reassignment = category_reassigment_frame.groupby(['Title', 'Artist'])['to'].agg(pd.Series.mode).reset_index()
        frame = pd.merge(frame, common_reassignment, on=['Title', 'Artist'], how='left')  # how='left' since not every song has been played perhaps
        print("Using perceived (re-assigned) categories for analysis.")
        frame['Category'] = frame['to']  # overwrite



        # plot category re-assignment as Sankey:
        _ = visualizations.plot_category_reassignment_sankey(
            category_reassignment_frame=category_reassigment_frame,
            song_colors=song_colors,
            show_title=False,
            output_dir=PLOT_DIR,
            rename_dict={'Classic':'Classical'},  # needs to be changed for visualizations
        )





    ### STANDARDIZATION
    if cluster_results:
        # standardize:
        scaler = StandardScaler()
        standardized_feature_array = scaler.fit_transform(feature_array)


    ### CLUSTERING
    if cluster_results:
        # parameters:
        visualize_umap: bool = True  # overwrites the below two
        plot_centroids: bool = True
        k = 4
        x_feature_ind: int = 0
        y_feature_ind: int = 3

        # compute k-means:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X=standardized_feature_array)
        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # prepare plots:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title("K-Means UMAP Visualization")

        # scatterplot:
        if visualize_umap:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            feature_embedding = umap_reducer.fit_transform(standardized_feature_array)
            centroid_embedding = umap_reducer.transform(centroids)

        # scatterplot:
        feature_x = feature_embedding[:, 0] if visualize_umap else standardized_feature_array[:, x_feature_ind]
        feature_y = feature_embedding[:, 1] if visualize_umap else standardized_feature_array[:, y_feature_ind]
        sc = ax.scatter(feature_x, feature_y, c=cluster_labels, cmap='Set1', s=15)

        if plot_centroids:
            centroid_x = centroid_embedding[:, 0] if visualize_umap else centroids[:, x_feature_ind]
            centroid_y = centroid_embedding[:, 1] if visualize_umap else centroids[:, x_feature_ind]
            sc_centroid = ax.scatter(centroid_x, centroid_y, c='black', marker='x', s=150, linewidths=3,
                                     label='Centroids')

        # legend:
        handles, lab_vals = sc.legend_elements()  # unique values from c
        lab_vals = [f"Cluster {ind}" for ind in lab_vals]
        if plot_centroids:
            handles += [sc_centroid]
            lab_vals += ['Centroids']

            x_min = min(np.min(feature_x), np.min(centroid_x))
            x_max = max(np.max(feature_x), np.max(centroid_x))
            ax.set_xlim(x_min - .05 * (x_max - x_min), x_max + .05 * (x_max - x_min))
            y_min = min(np.min(feature_y), np.min(centroid_y))
            y_max = max(np.max(feature_y), np.max(centroid_y))
            ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))


        ax.legend(handles, lab_vals)# title=category_label)#  title="Class")
        ax.set_xlabel('UMAP 1' if visualize_umap else feature_labels[x_feature_ind])
        ax.set_ylabel('UMAP 2' if visualize_umap else feature_labels[y_feature_ind])
        plt.show()


    ### MUTUAL INFORMATION
    if compute_mutual_information:

        if target_label == 'Genre': target_array = genres
        elif target_label == 'Category':  # remove familiarity label
            target_array = [cat.replace("Unfamiliar ", "").replace("Familiar ", "") for cat in categories]

        from src.pipeline.signal_features import compute_feature_mi_importance
        _, _ , feature_importance = compute_feature_mi_importance(feature_array, target_array,
                                                                  feature_labels, target_label, plot_save_dir=PLOT_DIR,)
        print(feature_importance)



    # analyse musical features via Scatter + KDE plots
    if plot_scatters:
        for x, y in scatter_x_y_combinations:
            # unpack if display label provided:
            if isinstance(x, tuple): x, x_label = x
            else: x_label = x
            if isinstance(y, tuple): y, y_label = y
            else: y_label = y

            x_ind = feature_labels.index(x)
            y_ind = feature_labels.index(y)
            if target_label == 'Genre':
                target_array = genres
            elif target_label == 'Category':  # remove familiarity label
                target_array = [cat.replace("Unfamiliar ", "").replace("Familiar ", "") for cat in categories]
            _ = visualizations.plot_scatter(x=feature_array[:, x_ind], y=feature_array[:, y_ind],
                                            x_label=x_label, y_label=y_label,
                                            category_list=target_array, category_label=target_label,
                                            cmap=list(song_colors.values()), save_dir=PLOT_DIR)
