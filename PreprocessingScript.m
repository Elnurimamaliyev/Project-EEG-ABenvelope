%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The script processes EEG and audio data, extracts features from the 
% audio (envelope, onset, binned amplitude), trains and evaluates models 
% using these features, and visualizes the results. % It includes steps 
% for integrating pre-processed data and performing feature extraction 
% and model training using the Temporal Response Function (TRF) framework.
% Ensure that you have a pre-processing pipeline set up to handle your raw 
% EEG and audio data before running this script.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Thorge Haupt.
% Modified by Elnur Imamaliyev. 
% Dataset: \\daten.uni-oldenburg.de\psychprojects$\Neuro\Thorge Haupt\data\Elnur
% Date: 04.09.2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning
clear, clc;
% Add Main paths
OT_setup

% % PRE-PROCESSING
% OT_preprocessing00_xdf_to_set
% OT_preprocessing00_xdf_to_set_modified
% OT_preprocessing0_ICA
% OT_preprocessing2_onsetwav

% Load the raw (unpreprocessed) data and Save as different variables
% Loop through subjects and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        %load the EEG data
        % EEG = []; 
        % Load the EEG data for each subject and task
        [EEG, PATH] = o2_preprocessing(s, k, sbj, 20);

        % Select the stimulus of interest (as named in the EEG.event structure)
        % stim_label = 'alarm_tone_post';
        % % Epoch the data with respect to the alarm tone
        % EEG_epoch = pop_epoch(EEG,{stim_label},[-1 2]) 
        % 
        % % Save the data for the current subject and task
        % temp_dat(:,:,:,k) = EEG_epoch.data;

        % Define the filename for saving the preprocessed data
        EEGFileName = sprintf('Preprocessed_EEG_Subjecttest_%s_Task_%s.mat', sbj{s}, task{k});
        % Save the preprocessed EEG data for each subject and task
        save(fullfile('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AllUsers\Preprocessed_Data\', EEGFileName), 'EEG');

    end
end

