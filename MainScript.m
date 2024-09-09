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
% o0_setupscript_trf
% o0_xdftoset_trf
% o1_preICA_trf
%% Load the raw (unpreprocessed) data
% EEG = pop_loadset('P001_ica_narrow_game_added_trigger.set ','C:\Users\icbmadmin\Documents\GitLabRep\PProject_TRFPP\TRFPP');
%% compute the ERP for the narow condition
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
%% Save relevant variables before proceeding
% save('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Relevant Variables\Relevant_Variables.mat', 'EEG'); clear;
%% Continue once you have a running pre-processing pipeline
load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Relevant Variables\Relevant_Variables.mat');    % Load the saved variables if needed
%% Continue once you have a running pre-processing pipeline
fs_eeg = EEG.srate;         % Get the final EEG sample rate
resp = EEG.data';           % Assign the EEG data to the resp variable

% Load the audio 
wavnarrow = load('P001_narrow_audio_strct.mat');
fs_audio = wavnarrow.audio_strct.srate; % Get the final audio sample rate
audio_dat = wavnarrow.audio_strct.data; % Assign the audio data to the audio_dat variable
%% FEATURE extraction 

%%% Original Envelope Feature Generation   
stim_Env = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox
stim_Env = stim_Env(1:size(resp, 1), :);                % Ensure the envelope matches the length of the EEG data

%%% ABenvelope Feature
num_bins = 8;

binEdges_dB = linspace(8, 64, num_bins + 1);

[stim_ABenv, NormEnv, NormBinEdges, BinEdges] = ABenvelopeGenerator_V1(stim_Env, num_bins);
% stim_Env = stim_Env';                                   % Transpose the envelope

%%% Onset Feature
stim_Onset = OnsetGenerator(stim_Env); 

%%% Combine Top matrix of ABenvelope with Onset
CmbOnsPlusTopABenv = stim_ABenv; CmbOnsPlusTopABenv(:,1) = stim_Onset;  % ABEnvelope and Onset Concatinated (BinNum, all)
%% Amplitude Dencity

% Count amplitude density of each bin (Using histcounts)
[DbCounts, ~, ~] = histcounts(NormEnv, NormBinEdges');

% Plot Histogram
figure;
histogram('BinEdges',binEdges_dB,'BinCounts',DbCounts);

% Title, labels and flip, add grid
set(gca,'view',[90 -90])
xlabel('Bin Edges (dB)'); ylabel('Counts');
title('Histogram of Frequency Counts');
grid on;

%% Ready Features
stim_Env;           % stim_Env - Original Envelope (1, all)
stim_ABenv;         % stim_ABenv - Amplitude-Binned Envelope (BinNum, all)
CmbOnsPlusTopABenv;    % Cmb_Ons_ABenvTop_C - ABEnvelope and Onset Concatinated (BinNum, all)

%% Using different features to get Accuracies
feature_names = {'Orig Env (norm)', 'ABenv', 'CmbOnsPlusTopABenv'};

% Add the binned envelope to the features list
features = {NormEnv, stim_ABenv, CmbOnsPlusTopABenv};

% Using different features to get Accuracies
figure;

col_num = length(features); % Dinamic Figure Columns


% Model hyperparameters
tmin = -100;
tmax = 400;
trainfold = 10;
testfold = 1;

for feature_idx = 1:length(features)

    stim = features{feature_idx};  % Assign the current feature set to stim
    fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set

    % Partition the data into training and test data segments
    [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);

    % Compute model weights
    model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);
    % Test the model on unseen neural data
    [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);

    % Plotting in a col_num x 2 grid

    % Model weights
    subplot(col_num, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(col_num, 2, feature_idx * 2)
    boxplot(stats.r) % channel correlation values
    title(sprintf('GFP (%s)', feature_names{feature_idx}))
    ylabel('correlation')
end

% Adjust the layout
sgtitle('Feature Comparisons') % Add a main title to the figure

%% V2 - Using different features to get Accuracies
feature_names = {'Orig Env (norm)', 'ABenv', 'CmbOnsPlusTopABenv'};

% Add the binned envelope to the features list
features = {NormEnv, stim_ABenv, CmbOnsPlusTopABenv};

% Using different features to get Accuracies
figure;

col_num = length(features); % Dinamic Figure Columns

% Model hyperparameters
tmin = -100;
tmax = 400;
% tmax = 500;
trainfold = 10;
testfold = 1;

Dir = 1; %specifies the forward modeling

lambdas = linspace(10e-4,10e4,10);

reg = cell(length(features), 1);
mod_w = cell(length(features), 1);

for feature_idx = 1:length(features)

    stim = features{feature_idx};  % Assign the current feature set to stim
    fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set



    % Partition the data into training and test data segments
    [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);


    %%% z-score the input and output data
    strainz = strain;
    stestz = stest;
    rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
    rtestz = zscore(rtest, [], 'all');


    %%% Use cross-validation to find the optimal regularization parameter
    fs = EEG.srate;
    cv = mTRFcrossval(strainz, rtrainz, fs, Dir, tmin, tmax, lambdas, 'Verbose', 0);

    %get the optimal regression parameter
    l = mean(cv.r,3); %over channels
    [l_val,l_idx] = max(mean(l,1));
    l_opt = lambdas(l_idx);
    
    % Train the neural model with the optimal regularization parameter
    model_train = mTRFtrain(strainz, rtrainz, fs, Dir, tmin, tmax, l_opt, 'verbose', 0);

    % Predict the neural data
    [PRED, STATS] = mTRFpredict(stestz, rtestz, model_train, 'verbose', 0);
    
    % Store results
    reg{feature_idx} = STATS.r;
    mod_w{feature_idx} = squeeze(model_train.w);

    % Plotting in a col_num x 2 grid

    % Model weights
    subplot(col_num, 2, feature_idx * 2 - 1)
    plot(model_train.t, squeeze(model_train.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(col_num, 2, feature_idx * 2)
    boxplot(STATS.r) % channel correlation values
    title(sprintf('GFP (%s)', feature_names{feature_idx}))
    ylabel('correlation')
end

% Adjust the layout
sgtitle('Feature Comparisons') % Add a main title to the figure

%% Plotting with zoom-in for correlation
% Extract and create time arrays
time_array = (0:wavnarrow.audio_strct.pnts-1) / wavnarrow.audio_strct.srate;  % Original time array (seconds)
time_array_stim = (0:length(stim_Env) - 1) / fs_eeg;  % Time array for the envelope (seconds)
%% Define the zoom-in range (in seconds)
start_window = 226;  % Start time in seconds
end_window = 242;     % End time in seconds
% Define the zoom-in range (in seconds)
% start_window = 233.5;  % Start time in seconds
% end_window = 235.5;     % End time in seconds
%% For plotting populate the matrix_bin with amplitude values
bins= 0:num_bins-1;
bins = bins(:)*8;

plot_stim_ABenv = repmat(bins, 1, length(stim_ABenv)); 
% plot_stim_ABenv = repmat(bins(:)*8, 1, length(stim_ABenv)); 
plot_stim_ABenv = plot_stim_ABenv' + stim_ABenv*4;

figure;

subplot(4,1,1);
plot(time_array_stim, resp)
title('EEG')
xlabel('Time (s)');
ylabel('Voltage (μV)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,2);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (s)');
ylabel('Amplitude (a.u)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,3);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (s)');
ylabel('Amplitude (a.u)')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Mark onsets on the envelope plot for better visualization
subplot(4,1,3);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, stim_Env(onset_indices), 'r');
hold off;
legend('Env', 'Ons');

subplot(4,1,4);
% plot((1:length(stim_binned))/fs_eeg, stim_binned_matrix);
plot((1:length(plot_stim_ABenv))/fs_eeg, plot_stim_ABenv);
title('AB Envelope');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);
%% Next
%%% Histogram Distribution 
addpath 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Thorge\AB_envelope'
% AB_envelope


%% Will not be used probably
% UnusedFeatures % Combine other features (e.g., )


%%
nums = [2 5 6 3 3 9 6 5 4 5 3]'


mask1 = [1 0 0 0 0 0 1 0 0 1 1]'
mask2 = [0 1 0 1 1 1 0 0 1 0 1]'

mask = [mask1, mask2]

neww = zeros(length(nums), 2);

for bin_idx = 1:2
    bin_mask = mask(:, bin_idx);
    % sum non-zero nums for differrent bins 

    neww(bin_mask==1, bin_idx) = nums(bin_mask==1)
end
%%
nums = [2 5 6 3 3 9 6 5 4 5 3]';

mask1 = [1 0 0 0 0 0 1 0 0 1 1]';
mask2 = [0 1 0 1 1 1 0 0 1 0 1]';

mask = [mask1, mask2];

neww = zeros(length(nums), 2);

for bin_idx = 1:2
    bin_mask = mask(:, bin_idx); % Use column for each bin's mask
    % sum non-zero nums for different bins 
    neww(bin_mask == 1, bin_idx) = nums(bin_mask == 1);
end

disp(neww);


