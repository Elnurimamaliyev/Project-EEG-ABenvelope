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
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\preprocessing') % Adding Preprocessing folder path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\audio')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\')
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
% %% Save relevant variables before proceeding
% save('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Relevant Variables\Relevant_Variables.mat', 'EEG');
%% Continue once you have a running pre-processing pipeline
% Load the saved variables if needed
load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Relevant Variables\Relevant_Variables.mat');
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

[stim_Env, stim_Onset] = EnvelopeGenerators(audio_dat, resp, fs_audio, fs_eeg);

% Combine features (e.g., envelope and onset)
% stim_combined = [stim_Env, stim_Onset];
% 
% % Ensure the stim_combined is the same length as resp
% stim_combined = stim_combined(1:size(resp,2),:);
% 
% % Now, stim_combined contains both the envelope and onset features aligned with EEG data.

%% Amplitude binned envelope
% functionAmplitude_binned_vol1

% Feature extraction and amplitude binning

% Define the number of bins for amplitude binning
num_bins = 16; % Number of bins, adjust as needed

% Compute the amplitude range for binning
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));
bin_edges = linspace(min_amp, max_amp, num_bins + 1);

% Initialize the binned envelope matrix
stim_binned = zeros(size(stim_Env));

% Bin the envelope data
for bin_idx = 1:num_bins
    bin_mask = (stim_Env >= bin_edges(bin_idx)) & (stim_Env < bin_edges(bin_idx + 1));
    stim_binned(bin_mask) = bin_idx;
end

% Convert the binned data to a binary matrix where each column corresponds to a bin
stim_binned_matrix = zeros(num_bins, length(stim_Env));

for bin_idx = 1:num_bins
    stim_binned_matrix(bin_idx,:) = (stim_binned == bin_idx);
end

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(:, 1:size(resp, 2));
%%
% stim = stim_binned;
% 
% 
% xobs = size(stim',1);
% yobs = size(resp',1);
% 
% % Check equal number of observations
% if ~isequal(xobs,yobs)
%     error(['STIM and RESP arguments must have the same number of '...
%         'observations.'])
% end

%% Using different features to get Accuracies
feature_names = {'Envelope', 'Onset', 'Amplitude-Binned Envelope'};

% Add the binned envelope to the features list
features = {stim_Env, stim_Onset, stim_binned_matrix};

% Using different features to get Accuracies
figure;

for feature_idx = 1:length(features)

    % Assign the current feature set to stim
    stim = features{feature_idx};

    % Optionally, display the name of the current feature set
    fprintf('Processing feature set: %s\n', feature_names{feature_idx});

    % Ensure the stim is the correct size (matching the length of resp)
    % stim = stim(:, 1:size(resp, 2));

    % Model hyperparameters
    tmin = -100;
    tmax = 400;
    trainfold = 10;
    testfold = 1;

    % Partition the data into training and test data segments
    [strain, rtrain, stest, rtest] = mTRFpartition(stim', resp', trainfold, testfold);

    % Compute model weights
    model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);

    % Test the model on unseen neural data
    [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);

    % Plotting in a 3x2 grid
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

%% Plotting with zoom-in for correlation

% Plotting the Envelope and Onset
% Extract and create time arrays
time_array = (0:wavnarrow.audio_strct.pnts-1) / wavnarrow.audio_strct.srate;  % Original time array (seconds)
time_array_stim = (0:length(stim_Env) - 1) / fs_eeg;  % Time array for the envelope (seconds)

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds


%%

% Populate the matrix_bin with amplitude values
bins= 0:num_bins-1;
plot_matrix_bin = repmat(bins(:), 1, length(stim_binned)); 
plot_matrix_bin = plot_matrix_bin + stim_binned_matrix;

%%
figure;
subplot(3,1,1);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (s)');
ylabel('Amplitude (db)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(3,1,2);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (s)');
ylabel('Envelope Amplitude')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Mark onsets on the envelope plot for better visualization
subplot(3,1,2);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, stim_Env(onset_indices), 'r');
legend('Envelope', 'Onsets');
hold off;

% Mark onsets on the envelope plot for better visualization
subplot(3,1,3);
% plot((1:length(stim_binned))/fs_eeg, stim_binned_matrix);
plot((1:length(plot_matrix_bin))/fs_eeg, plot_matrix_bin);
title('Binned stim');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);
ylim([-2 17]);

%% (!) Here



%%% Logarithmic Binning
% Define dB bin edges (in dB)
db_bin_edges = [8, 16, 24, 32, 40, 48, 56, 64]; % Example edges in dB

% Convert dB bin edges to linear scale
linear_bin_edges = 10 .^ (db_bin_edges / 20);

% Normalize bin edges to the range [0, 1]
linear_bin_edges = (linear_bin_edges - min(linear_bin_edges)) / (max(linear_bin_edges) - min(linear_bin_edges));

% Bin the envelope data using histcounts
[~, bin_indices] = histcounts(stim_Env, linear_bin_edges);

% Initialize the binned envelope matrix
stim_binned_matrix = zeros(size(stim_Env, 1), length(db_bin_edges));

% Bin the envelope data
for bin_idx = 1:length(db_bin_edges)
    % Create a binary matrix where each column corresponds to a bin
    stim_binned_matrix(bin_idx,:) = (bin_indices == bin_idx);
end

% Normalize each bin column to range [0, 1]
stim_binned_matrix = stim_binned_matrix ./ max(stim_binned_matrix, [], 1);

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(1:size(resp, 2), :);



%% Modification for numbers in beetween
amp_conti = [2, 3.2, 4.1, 0.1, 3.3, 4.4, 5.5, 44.5, 0.6, 8, 0, 3, 0, 0.6, 4.3, 5, 1, 3, 9, 5, 7, ...
    6, 1, 8, 8.5, 11, 2, 13.5, 2, 7, 12, 12, 12, 11, 10];

bins = 0:8:64;
%%
load('short_audio.mat')
load('short_envelope.mat')

audio = short_audio;
amp_conti = short_envelope';
bins = [0, 8, 16, 24, 32, 40, 48, 56, 64];
%%

[plot_matrix_bin, matrix_bin] = ABenvelope(amp_conti, bins);

% identify max and min value
min_val = min(amp_conti); max_val = max(bins);

% time array
t = 1:length(amp_conti);

% Plot Modified Amplitude Levels
% figure;
subplot(3, 1, 1);
plot(t, audio);
title('Amplitude over Time');
xlabel('Time');
ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);

subplot(3, 1, 2);
plot(t, amp_conti);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);

subplot(3, 1, 3);
plot(t, plot_matrix_bin);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([min_val-1, max_val+4]);

%% function
function [plot_matrix_bin, matrix_bin] = ABenvelope(amp_conti, bins)
    % Initialize variables
    matrix_bin = zeros(length(bins), length(amp_conti));
    % Update the matrix_bin based on the given logic
    for step_counter = 1:length(amp_conti)
        amp_value = round(amp_conti(step_counter));         % Round amplitude value to nearest integer
        [~, closest_bin_idx] = min(abs(bins - amp_value));  % Find the closest bin to the amplitude value
        matrix_bin(closest_bin_idx, step_counter) = matrix_bin(closest_bin_idx, step_counter) + 1; % Add 1 to the current amplitude value
    end

    % Populate the matrix_bin with amplitude values
    plot_matrix_bin = repmat(bins(:), 1, length(amp_conti)); 
    plot_matrix_bin = plot_matrix_bin + matrix_bin;
end


