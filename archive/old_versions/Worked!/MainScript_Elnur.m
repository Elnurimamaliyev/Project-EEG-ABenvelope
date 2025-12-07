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
time_array_stim = (0:length(stim_Env) - 1) / fs_eeg;  % Time array for the envelope (seconds)

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
plot(time_array_stim, stim_Env)
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

%% Amplitude binned envelope



% Feature extraction and amplitude binning

% Define the number of bins for amplitude binning
num_bins = 10; % Number of bins, adjust as needed

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
stim_binned_matrix = zeros(size(stim_Env, 1), num_bins);

for bin_idx = 1:num_bins
    stim_binned_matrix(:, bin_idx) = (stim_binned == bin_idx);
end

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(1:size(resp, 2), :);

% Update feature names
feature_names = {'Envelope', 'Amplitude-Binned Envelope'};

% Add the binned envelope to the features list
features = {stim_Env, stim_binned_matrix};

% Using different features to get Accuracies
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

    % Plotting in a 3x2 grid
    % Model weights
    subplot(2, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(2, 2, feature_idx * 2)
    boxplot(stats.r) % channel correlation values
    title(sprintf('GFP (%s)', feature_names{feature_idx}))
    ylabel('correlation')
end

% Adjust the layout
sgtitle('Feature Comparisons') % Add a super title to the figure


%%

% Plotting with zoom-in for correlation
figure;
subplot(4,1,1);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (seconds)')
ylabel('Amplitude')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,2);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (seconds)')
ylabel('Envelope Amplitude')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Plot the detected onsets
subplot(4,1,3);
plot((1:length(stim_Onset))/fs_eeg, stim_Onset);
title('Detected Onsets');
xlabel('Time (s)');
ylabel('Onset Detection');
xlim([start_window end_window]);  

% Mark onsets on the envelope plot for better visualization
subplot(4,1,4);
plot((1:length(stim_Onset))/fs_eeg, stim_binned);
title('Binned stim');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);  


%%
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

%%

% Define time arrays
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
time_env = (0:length(stim_Env) - 1) / fs_eeg;       % Time array for envelope and binned envelope

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds

% Plotting time series
figure;

% Plot original sound
subplot(3, 1, 1);
plot(time_audio, audio_dat);
title('Original Sound');
xlabel('Time (seconds)');
ylabel('Amplitude');
% xlim([start_window end_window]);  % Zoom in on the original sound

% Plot envelope
subplot(3, 1, 2);
plot(time_env, stim_Env);
title('Envelope');
xlabel('Time (seconds)');
ylabel('Envelope Amplitude');
% xlim([start_window end_window]);  % Zoom in on the envelope

% Plot amplitude-binned envelope
subplot(3, 1, 3);
imagesc(time_env, db_bin_edges, stim_binned_matrix'); % Transpose for correct orientation
% axis xy; % Correct orientation of y-axis
title('Amplitude-Binned Envelope');
xlabel('Time (seconds)');
ylabel('Amplitude Bin');
% xlim([start_window end_window]);  % Zoom in on the envelope
% colorbar; % Add colorbar for bin representation

% Adjust the layout
sgtitle('Time Series of Original Sound, Envelope, and Amplitude-Binned Envelope'); % Add a super title to the figure


% Define time arrays
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
time_env = (0:length(stim_Env) - 1) / fs_eeg;       % Time array for envelope and binned envelope

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds

% Plotting time series
figure;

% Plot original sound
subplot(3, 1, 1);
plot(time_audio, audio_dat);
title('Original Sound');
xlabel('Time (seconds)');
ylabel('Amplitude');
xlim([start_window end_window]);  % Zoom in on the original sound

% Plot envelope
subplot(3, 1, 2);
plot(time_env, stim_Env);
title('Envelope');
xlabel('Time (seconds)');
ylabel('Envelope Amplitude');
xlim([start_window end_window]);  % Zoom in on the envelope

% Plot amplitude-binned envelope
subplot(3, 1, 3);
imagesc(time_env, 1:length(db_bin_edges), stim_binned_matrix'); % Transpose for correct orientation
axis xy; % Correct orientation of y-axis
title('Amplitude-Binned Envelope');
xlabel('Time (seconds)');
ylabel('Amplitude Bin');
xlim([start_window end_window]);  % Zoom in on the envelope
% colorbar; % Add colorbar for bin representation

% Adjust the layout
sgtitle('Time Series of Original Sound, Envelope, and Amplitude-Binned Envelope'); % Add a super title to the figure
%%%



%%

amp_conti = [1, 1, 6, 1, 4, 5, 6, 6, 8, 2, 0,  3, 1, 8, 2, 4, 5, 1, 8, 5, 7, ...
    6, 1, 8, 8, 6, 2, 4, 2, 7, 11, 8, 12, 11, 10];
t = 1:length(amp_conti);

    % histcounts
    [~, bin_indices] = histcounts(amp_conti, min_val:max_val);

    % Initialize variables
    t = 1:length(amp_conti);
    matrix_bin = zeros(length(bin_indices), length(t)); 

    % Populate the matrix_bin with amplitude values
    for i = 1:length(bin_indices)
        amp_value = amp_conti(i);
        matrix_bin(i, 1:end) = i; % Pre-fill each row with its amplitude level
    end

%%
amp_conti = [1, 1, 6, 1, 4, 5, 6, 6, 8, 2, 0,  3, 1, 8, 2, 4, 5, 1, 8, 5, 7, ...
    6, 1, 8, 8, 6, 2, 4, 2, 7, 11, 8, 12, 11, 10];
t = 1:length(amp_conti);

% Define bin width
bin_width = 2;

% Determine bin edges
min_val = floor(min(amp_conti) / bin_width) * bin_width;
max_val = ceil(max(amp_conti) / bin_width) * bin_width;
bin_edges = min_val:bin_width:max_val;

% Compute histogram counts and bin indices
[~, bin_indices] = histcounts(amp_conti, bin_edges);

% Initialize the matrix
num_bins = length(bin_edges) - 1;
matrix_bin = zeros(num_bins, length(amp_conti));

% Populate the matrix
for i = 1:length(amp_conti)
    % Determine bin index for current amplitude value
    bin_idx = bin_indices(i);
    
    % Check if bin_idx is valid
    if bin_idx > 0 && bin_idx <= num_bins
        matrix_bin(bin_idx, i) = amp_conti(i);
    end
end

% Display the matrix (optional)
disp('Matrix Bin:');
disp(matrix_bin);


%%

amp_conti = [1, 1, 6, 1, 4, 5, 6, 6, 8, 2, 0,  3, 1, 8, 2, 4, 5, 1, 8, 5, 7, ...
    6, 1, 8, 8, 6, 2, 4, 2, 7, 11, 8, 12, 11, 10];
t = 1:length(amp_conti);

% Define bin width
bin_width = 1;

% Determine bin edges
min_val = floor(min(amp_conti) / bin_width) * bin_width;
max_val = ceil(max(amp_conti) / bin_width) * bin_width;
bin_edges = min_val-bin_width:bin_width:max_val+bin_width;

% Compute histogram counts and bin indices
[~, bin_indices] = histcounts(amp_conti, bin_edges);

% Initialize the matrix
num_bins = length(bin_edges) - 1;

% Initialize the matrix to zeros
matrix_bin = zeros(num_bins, length(amp_conti));

% Fill the matrix
for i = 1:length(amp_conti)

    bin_value = bin_indices(i); % Get the bin value

    find()

    matrix_bin(bin_index, i) = bin_value; % Place the bin value in the matrix
end




%%
% Populate the matrix
for i = 1:length(amp_conti)
    % bin_idx =     % Determine bin index for current amplitude value
    
    % Check if bin_idx is valid
    if bin_idx >= 0 && bin_idx < num_bins
        matrix_bin(bin_idx + 1, i) = amp_conti(i);
    end
end

% Display the matrix (optional)
disp('Matrix Bin:');
disp(matrix_bin);

%%

    % Update the matrix_bin based on the given logic
    for i = 1:length(amp_conti)
        amp_value = amp_conti(i);

        matrix_bin(amp_value, i) = matrix_bin(amp_value, i) + 0.5; % Add 0.5 to the current amplitude value
    end

%%
% end

% matrix_bin = AmplitudeBinner(amp_conti);


% Create the first figure for the original amplitude data
figure;
subplot(2,1,1);
plot(t, amp_conti);
title('Amplitude over Time');
xlabel('Time');
ylabel('Amplitude');

% Create the second figure for the modified amplitude levels
subplot(2,1,2);
plot(t, matrix_bin);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');






























%%
amp_conti = [1, 4, 5, 6, 6, 8, 2, 3, 1, 8, 2, 4, 5, 1, 8, 5, 7, ...
    6, 1, 8, 8, 6, 2, 4, 2, 7, 9, 8, 3, 4];
t = 1:length(amp_conti);


matrix_bin = AmplitudeBinner(amp_conti)








% 
% % Initialize the matrix to store the modified amplitude values for each level
% matrix_bin = zeros(10, length(t)); % 10 rows for 10 amplitudes, columns for time points
% 
% % Populate the matrix_bin with amplitude values
% for i = 1:10
%     matrix_bin(i, :) = i; % Pre-fill each row with its amplitude level
% end
% 
% % Update the matrix_bin based on the given logic
% for i = 1:length(amp_conti)
%     amp_value = amp_conti(i);
%     if amp_value >= 1 && amp_value <= 10 % Ensure amplitude value is within the valid range
%         matrix_bin(amp_value, i) = matrix_bin(amp_value, i) + 0.5; % Add 0.5 to the current amplitude value
%     end
% end

% Create the first figure for the original amplitude data
figure;
subplot(2,1,1);
plot(t, amp_conti);
title('Amplitude over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([0 10]); % Set y-axis limits

% Create the second figure for the modified amplitude levels
subplot(2,1,2);
plot(t, matrix_bin);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([0 10]); % Set y-axis limits

% Create a legend for the amplitudes
legend(arrayfun(@(x) sprintf('Amplitude %d', x), 1:10, 'UniformOutput', false));





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


