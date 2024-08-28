%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script shows you how to apply the TRF toolbox to your data
% a prerequisite is that you have a working pre-processing pipeline which
% you can integrate here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Thorge Haupt.
% Modified by Elnur Imamaliyev. 
% Date: 27.08.2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning
clear, clc;
%% Turn off the warnings
%#ok<*SAGROW>
%#ok<*NASGU>
%#ok<*NOPTS>

%% Add Main path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\preprocessing') % Adding Preprocessing folder path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Data\audio')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\mTRF-Toolbox\mtrf\')

% %% PRE-PROCESSING
% o0_setupscript_trf
% o0_xdftoset_trf
% o1_preICA_trf
% 
% %% Load the raw (unpreprocessed) data
% % EEG = pop_loadset('P001_ica_narrow_game_added_trigger.set ','C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\TRFPP');
% %%
% %compute the ERP for the two conditions separately
% for k = 1:length(task)
%     %load the EEG data
%     EEG = []; 
%     [EEG,PATH] = o2_preprocessing(1,k,sbj,20);
% 
%     %select the stimulus of interest (as named in the EEG.event structure)
%     stim_label = 'alarm_tone_post';
% 
%     %epoch the data with respect to the alarm tone
%     EEG_epoch = pop_epoch(EEG,{stim_label},[-1 2]) 
% 
%     %save the data 
%     temp_dat(:,:,:,k) = EEG_epoch.data; 
% end
% 
% 
% %% Save relevant variables before proceeding
% save('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Relevant Variables\Relevant_Variables.mat', 'EEG');

%% Continue once you have a running pre-processing pipeline
% Load the saved variables if needed
load('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Relevant Variables\Relevant_Variables.mat');
%% Continue once you have a running pre-processing pipeline
% Get the final EEG srate
fs_eeg = EEG.srate;

% Assign the EEG data to the resp variable
resp = EEG.data;

% Load the audio 
wavnarrow = load('P001_narrow_audio_strct.mat');
fs_audio = wavnarrow.audio_strct.srate;
audio_dat = wavnarrow.audio_strct.data;

%% FEATURE extraction 

%%% Envelope Generation
% We extract the envelope
mEnv = mTRFenvelope(double(audio_dat)',fs_audio,fs_eeg); % in order to use the mTRF toolbox the eeg data and stimulus need to be the same length 
% i.e. the same sample rate

% Assign to the stimulus variable
stim_Env = mEnv;

% Are stim and resp the same length?
stim_Env = stim_Env(1:size(resp,2),:);

%%% Onset Generator
% Resample the audio data to match the EEG sampling rate if not already done
% if fs_audio ~= fs_eeg
%     audio_dat_res = resample(audio_dat, fs_eeg, fs_audio);
% end

% Threshold-based peak detection on the envelope
% Adjust the threshold and minimum peak distance based on your data
threshold = 0.3; % Set a threshold for peak detection
min_peak_distance = 0.15 * fs_eeg; % Minimum distance between peaks in samples

[onset_peaks, onset_locs] = findpeaks(mEnv, 'MinPeakHeight', threshold, 'MinPeakDistance', min_peak_distance);

% Create onset feature vector
onsets = zeros(size(mEnv));
onsets(onset_locs) = 1;

% Trim or pad onset feature to match the length of the EEG data
stim_Onset = onsets(1:size(resp,2));


%%% Plotting the Envelope and Onset

% Extract and create time arrays
time_array = (0:wavnarrow.audio_strct.pnts-1) / wavnarrow.audio_strct.srate;  % Original time array (seconds)
time_array_stim = (0:length(stim) - 1) / fs_eeg;  % Time array for the envelope (seconds)

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds

% Plotting with zoom-in for correlation
figure;
subplot(3,1,1);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (seconds)')
ylabel('Amplitude')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(3,1,2);
plot(time_array_stim, stim)
title('Envelope')
xlabel('Time (seconds)')
ylabel('Envelope Amplitude')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Plot the detected onsets
subplot(3,1,3);
plot((1:length(stim_Onset))/fs_eeg, stim_Onset);
title('Detected Onsets');
xlabel('Time (s)');
ylabel('Onset Detection');
xlim([start_window end_window]);  

% Mark onsets on the envelope plot for better visualization
subplot(3,1,2);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, mEnv(onset_indices), 'r');
legend('Envelope', 'Onsets');
hold off;



%% Combine features (e.g., envelope and onset)
stim_combined = [stim_Env, stim_Onset];

% Ensure the stim_combined is the same length as resp
stim_combined = stim_combined(1:size(resp,2),:);

% Now, stim_combined contains both the envelope and onset features aligned with EEG data.

%% Using different features to get Accuracies
features = {stim_Env, stim_Onset, stim_combined};
feature_names = {'Envelope', 'Onset', 'Envelope+Onset'}; % Names for features

% Create a new figure for all subplots
figure;

for feature_idx = 1:length(features)
    
    % Assign the current feature set to stim
    stim = features{feature_idx};
    
    % Optionally, display the name of the current feature set
    fprintf('Processing feature set: %s\n', feature_names{feature_idx});
    
    % Ensure the stim is the correct size (matching the length of resp)
    stim = stim(1:size(resp, 2), :);

    % Model hyperparameters
    tmin = -100;
    tmax = 400;
    trainfold = 10;
    testfold = 1;
    
    % Partition the data into training and test data segments
    [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp', trainfold, testfold);

    % Compute model weights
    model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);
        
    % Test the model on unseen neural data
    [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);
    
    % Plotting in a 2x3 grid
    % Model weights
    subplot(3, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')
    
    % GFP (Global Field Power)
    subplot(3, 2, feature_idx * 2)
    boxplot(stats.r) % channel correlation values
    title(sprintf('GFP (%s)', feature_names{feature_idx}))
    ylabel('correlation')
end

% Adjust the layout
sgtitle('Feature Comparisons') % Add a super title to the figure

