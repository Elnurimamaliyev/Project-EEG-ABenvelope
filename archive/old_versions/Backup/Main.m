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
% Date: 04.09.2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning
clear, clc;
%% Add Main path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\preprocessing') % Adding Preprocessing folder path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\audio')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\mTRF-Toolbox\mtrf\')
%% PRE-PROCESSING
o0_setupscript_trf
o0_xdftoset_trf
o1_preICA_trf
%% Load the raw (unpreprocessed) data
% EEG = pop_loadset('P001_ica_narrow_game_added_trigger.set ','C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\TRFPP');
%% compute the ERP for the narow condition
for k = 1:length(task)
    %load the EEG data
    EEG = []; 
    [EEG,PATH] = o2_preprocessing(1,k,sbj,20);

    %select the stimulus of interest (as named in the EEG.event structure)
    stim_label = 'alarm_tone_post';

    %epoch the data with respect to the alarm tone
    EEG_epoch = pop_epoch(EEG,{stim_label},[-1 2]) 

    %save the data 
    temp_dat(:,:,:,k) = EEG_epoch.data; 
end

%% Save relevant variables before proceeding
% save('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Relevant Variables\Relevant_Variables.mat', 'EEG');
%% Continue once you have a running pre-processing pipeline
% Load the saved variables if needed
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Relevant Variables\Relevant_Variables.mat');
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

% Envelope Feature Generation   
stim_Env = mTRFenvelope(double(audio_dat), fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox
% Ensure the envelope matches the length of the EEG data
stim_Env = stim_Env(1:size(resp, 2), :);
% Transpose the envelope
stim_Env = stim_Env';

% Onset Feature
stim_Onset = OnsetGenerator(stim_Env'); 

% Onset Envelope (Onset + Envelope)
Onset_Envelope = stim_Env+ stim_Onset/8; % Dividing to onset envelope to fractions to just emphasize onsets in Envelope

% num_bins = 16;
% stim_binned_matrix = ABenvelopeGenerator_V1(stim_Env, num_bins);
% 
% % Ensure the binned matrix is the same length as resp
% stim_binned_matrix = stim_binned_matrix(:, 1:size(resp, 2));
% For plotting populate the matrix_bin with amplitude values
%
% bins= 0:num_bins-1;
% plot_matrix_bin = repmat(bins(:), 1, length(stim_binned_matrix)); 
% plot_matrix_bin = plot_matrix_bin + stim_binned_matrix;

%%
num_bins = 8; % Number of bins, adjust as needed

% Compute the amplitude range for binning
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));

% bin_edges = linspace(min_amp, max_amp, num_bins + 1);
bin_edges = logspace(log10(min_amp), log10(max_amp), num_bins + 1);

% Initialize the binned envelope matrix
stim_binned = zeros(num_bins,size(stim_Env,2));

% Bin the envelope data
for bin_idx = 1:num_bins
    bin_mask = (stim_Env >= bin_edges(bin_idx)) & (stim_Env < bin_edges(bin_idx + 1));
    % sum non-zero nums for differrent bins 
    % stim_binned(bin_mask) = bin_idx;
    stim_binned(bin_idx, bin_mask) = stim_Env(1,bin_mask);
end
stim_ABenv = normalize(stim_binned,2,'range');


% Convert the binned data to a binary matrix where each column corresponds to a bin
% stim_ABenv = zeros(num_bins, length(stim_Env));

% for bin_idx = 1:num_bins
%     stim_ABenv(bin_idx,:) = (stim_binned == bin_idx);
% end




%% Combine ABenvelope with Onset
%%% Ready
% stim_Env - Normal Envelope
% stim_ABenv - Amplitude-Binned Envelope

Onset23x_ABenvelope = stim_ABenv + stim_Onset*23;

% Normalize each bin to have values between 0 and 1
Normalized_Onset20x_ABenvelope = Onset23x_ABenvelope ./ max(Onset23x_ABenvelope, [], 2);

%%%
% Normalize each bin to have values between 0 and 64, with bin widths of 8
% Scale the values so that each bin corresponds to a range of 8
% More_Normalized_Onset20x_ABenvelope = Normalized_Onset20x_ABenvelope * 8;
% 
% More_Normalized_Onset20x_ABenvelope = More_Normalized_Onset20x_ABenvelope + stim_Env*2;
% 

%%% Using different features to get Accuracies
feature_names = {'Env', 'Onset', 'ABenv', '23XOnset+ABEnvelope', 'Normalised Onset+ABEnvelope'};

% Add the binned envelope to the features list
features = {stim_Env, stim_Onset, stim_ABenv, Onset23x_ABenvelope, Normalized_Onset20x_ABenvelope};

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

    % ColumnNum = length(features);

    % Plotting in a 3x2 grid
    % Model weights
    subplot(5, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(5, 2, feature_idx * 2)
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
% start_window = 226;  % Start time in seconds
% end_window = 242;     % End time in seconds
% Define the zoom-in range (in seconds)
start_window = 233.5;  % Start time in seconds
end_window = 235.5;     % End time in seconds


%% For plotting populate the matrix_bin with amplitude values
bins= 0:num_bins-1;
plot_stim_ABenv = repmat(bins(:)*2, 1, length(stim_ABenv)); 
plot_stim_ABenv = plot_stim_ABenv + stim_ABenv;

%%
figure;
subplot(4,1,1);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (s)');
ylabel('Amplitude (db)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,2);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (s)');
ylabel('Envelope Amplitude')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Mark onsets on the envelope plot for better visualization
subplot(4,1,2);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, stim_Env(onset_indices), 'r');
hold off;
legend('Env', 'Ons');

subplot(4,1,3);
% plot((1:length(stim_binned))/fs_eeg, stim_binned_matrix);
plot((1:length(plot_stim_ABenv))/fs_eeg, plot_stim_ABenv);
title('AB Envelope');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);

subplot(4,1,4);
plot(time_array_stim, Onset_Envelope)
title('Onset+Envelope')
xlabel('Time (s)');
ylabel('Amplitude (db)')
xlim([start_window end_window]);  % Zoom in on the original sound
legend('Env+Ons');



%% Next
% Combine features (e.g., envelope and onset)

%% Logarithmic Binning

% stim_Env

% Convert amplitude to decibels
stim_Env_dB = 20 * log10(stim_Env);

% Define the bin width in dB
dB_bin_width = 8; % Bin width in dB

% Determine the bin edges in dB
bin_edges_dB = 0:dB_bin_width:24; % Adjust as needed based on your data range
bin_edges_dB = [bin_edges_dB, max(stim_Env_dB)]; % Ensure the last bin edge covers the maximum value

% Convert bin edges from dB to amplitude ratios
bin_edges_amp = 10.^(bin_edges_dB / 20);

% Normalize the amplitude range to [0, 1]
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));
stim_Env_normalized = (stim_Env - min_amp) / (max_amp - min_amp);

% Initialize the binned envelope matrix
stim_binned = zeros(size(stim_Env));

% Bin the envelope data based on the normalized amplitude
for bin_idx = 1:numel(bin_edges_amp) - 1
    bin_mask = (stim_Env_normalized >= (bin_edges_amp(bin_idx) - min_amp) / (max_amp - min_amp)) & ...
               (stim_Env_normalized < (bin_edges_amp(bin_idx + 1) - min_amp) / (max_amp - min_amp));
    stim_binned(bin_mask) = bin_idx;
end

% Convert the binned data to a binary matrix where each row corresponds to a bin
stim_binned_matrix = zeros(numel(bin_edges_dB) - 1, length(stim_Env));

for bin_idx = 1:numel(bin_edges_dB) - 1
    stim_binned_matrix(bin_idx, :) = (stim_binned == bin_idx);
end

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(:, 1:size(resp, 2));












%%%%%%%%
% 2
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
% Date: 04.09.2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning
clear, clc;
%% Add Main path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\preprocessing') % Adding Preprocessing folder path
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\audio')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\')
addpath('C:\Users\icbmadmin\Documents\GitLabRep\mTRF-Toolbox\mtrf\')
%% PRE-PROCESSING
o0_setupscript_trf
o0_xdftoset_trf
o1_preICA_trf
%% Load the raw (unpreprocessed) data
% EEG = pop_loadset('P001_ica_narrow_game_added_trigger.set ','C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\TRFPP');
%% compute the ERP for the narow condition
for k = 1:length(task)
    %load the EEG data
    EEG = []; 
    [EEG,PATH] = o2_preprocessing(1,k,sbj,20);

    %select the stimulus of interest (as named in the EEG.event structure)
    stim_label = 'alarm_tone_post';

    %epoch the data with respect to the alarm tone
    EEG_epoch = pop_epoch(EEG,{stim_label},[-1 2]) 

    %save the data 
    temp_dat(:,:,:,k) = EEG_epoch.data; 
end

%% Save relevant variables before proceeding
% save('C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\Relevant Variables\Relevant_Variables.mat', 'EEG');
%% Continue once you have a running pre-processing pipeline
% Load the saved variables if needed
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Relevant Variables\Relevant_Variables.mat');
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

% Envelope Feature Generation   
stim_Env = mTRFenvelope(double(audio_dat), fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox
% Ensure the envelope matches the length of the EEG data
stim_Env = stim_Env(1:size(resp, 2), :);
% Transpose the envelope
stim_Env = stim_Env';

% Onset Feature
stim_Onset = OnsetGenerator(stim_Env'); 

% Onset Envelope (Onset + Envelope)
Onset_Envelope = stim_Env+ stim_Onset/8; % Dividing to onset envelope to fractions to just emphasize onsets in Envelope

% num_bins = 16;
% stim_binned_matrix = ABenvelopeGenerator_V1(stim_Env, num_bins);
% 
% % Ensure the binned matrix is the same length as resp
% stim_binned_matrix = stim_binned_matrix(:, 1:size(resp, 2));
% For plotting populate the matrix_bin with amplitude values
%
% bins= 0:num_bins-1;
% plot_matrix_bin = repmat(bins(:), 1, length(stim_binned_matrix)); 
% plot_matrix_bin = plot_matrix_bin + stim_binned_matrix;

%%
num_bins = 8; % Number of bins, adjust as needed

% Compute the amplitude range for binning
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));

% bin_edges = linspace(min_amp, max_amp, num_bins + 1);
bin_edges = logspace(log10(min_amp), log10(max_amp), num_bins + 1);

% Initialize the binned envelope matrix
stim_binned = zeros(num_bins,size(stim_Env,2));

% Bin the envelope data
for bin_idx = 1:num_bins
    bin_mask = (stim_Env >= bin_edges(bin_idx)) & (stim_Env < bin_edges(bin_idx + 1));
    % sum non-zero nums for differrent bins 
    % stim_binned(bin_mask) = bin_idx;
    stim_binned(bin_idx, bin_mask) = stim_Env(1,bin_mask);
end
stim_ABenv = normalize(stim_binned,2,'range');


% Convert the binned data to a binary matrix where each column corresponds to a bin
% stim_ABenv = zeros(num_bins, length(stim_Env));

% for bin_idx = 1:num_bins
%     stim_ABenv(bin_idx,:) = (stim_binned == bin_idx);
% end




%% Combine ABenvelope with Onset
%%% Ready
% stim_Env - Normal Envelope
% stim_ABenv - Amplitude-Binned Envelope

Onset23x_ABenvelope = stim_ABenv + stim_Onset*23;

% Normalize each bin to have values between 0 and 1
Normalized_Onset20x_ABenvelope = Onset23x_ABenvelope ./ max(Onset23x_ABenvelope, [], 2);

%%%
% Normalize each bin to have values between 0 and 64, with bin widths of 8
% Scale the values so that each bin corresponds to a range of 8
% More_Normalized_Onset20x_ABenvelope = Normalized_Onset20x_ABenvelope * 8;
% 
% More_Normalized_Onset20x_ABenvelope = More_Normalized_Onset20x_ABenvelope + stim_Env*2;
% 

%%% Using different features to get Accuracies
feature_names = {'Env', 'Onset', 'ABenv', '23XOnset+ABEnvelope', 'Normalised Onset+ABEnvelope'};

% Add the binned envelope to the features list
features = {stim_Env, stim_Onset, stim_ABenv, Onset23x_ABenvelope, Normalized_Onset20x_ABenvelope};

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

    % ColumnNum = length(features);

    % Plotting in a 3x2 grid
    % Model weights
    subplot(5, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(5, 2, feature_idx * 2)
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
% start_window = 226;  % Start time in seconds
% end_window = 242;     % End time in seconds
% Define the zoom-in range (in seconds)
start_window = 233.5;  % Start time in seconds
end_window = 235.5;     % End time in seconds


%% For plotting populate the matrix_bin with amplitude values
bins= 0:num_bins-1;
plot_stim_ABenv = repmat(bins(:)*2, 1, length(stim_ABenv)); 
plot_stim_ABenv = plot_stim_ABenv + stim_ABenv;

%%
figure;
subplot(4,1,1);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (s)');
ylabel('Amplitude (db)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,2);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (s)');
ylabel('Envelope Amplitude')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Mark onsets on the envelope plot for better visualization
subplot(4,1,2);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, stim_Env(onset_indices), 'r');
hold off;
legend('Env', 'Ons');

subplot(4,1,3);
% plot((1:length(stim_binned))/fs_eeg, stim_binned_matrix);
plot((1:length(plot_stim_ABenv))/fs_eeg, plot_stim_ABenv);
title('AB Envelope');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);

subplot(4,1,4);
plot(time_array_stim, Onset_Envelope)
title('Onset+Envelope')
xlabel('Time (s)');
ylabel('Amplitude (db)')
xlim([start_window end_window]);  % Zoom in on the original sound
legend('Env+Ons');



%% Next
% Combine features (e.g., envelope and onset)

%% Logarithmic Binning

% stim_Env

% Convert amplitude to decibels
stim_Env_dB = 20 * log10(stim_Env);

% Define the bin width in dB
dB_bin_width = 8; % Bin width in dB

% Determine the bin edges in dB
bin_edges_dB = 0:dB_bin_width:24; % Adjust as needed based on your data range
bin_edges_dB = [bin_edges_dB, max(stim_Env_dB)]; % Ensure the last bin edge covers the maximum value

% Convert bin edges from dB to amplitude ratios
bin_edges_amp = 10.^(bin_edges_dB / 20);

% Normalize the amplitude range to [0, 1]
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));
stim_Env_normalized = (stim_Env - min_amp) / (max_amp - min_amp);

% Initialize the binned envelope matrix
stim_binned = zeros(size(stim_Env));

% Bin the envelope data based on the normalized amplitude
for bin_idx = 1:numel(bin_edges_amp) - 1
    bin_mask = (stim_Env_normalized >= (bin_edges_amp(bin_idx) - min_amp) / (max_amp - min_amp)) & ...
               (stim_Env_normalized < (bin_edges_amp(bin_idx + 1) - min_amp) / (max_amp - min_amp));
    stim_binned(bin_mask) = bin_idx;
end

% Convert the binned data to a binary matrix where each row corresponds to a bin
stim_binned_matrix = zeros(numel(bin_edges_dB) - 1, length(stim_Env));

for bin_idx = 1:numel(bin_edges_dB) - 1
    stim_binned_matrix(bin_idx, :) = (stim_binned == bin_idx);
end

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(:, 1:size(resp, 2));
