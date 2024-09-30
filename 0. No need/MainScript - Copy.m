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

% PRE-PROCESSING
% OT_setup
% OT_preprocessing00_xdf_to_set
% OT_preprocessing00_xdf_to_set_modified
% OT_preprocessing0_ICA
% OT_preprocessing2_onsetwav

%% Load the raw (unpreprocessed) data and Save as different variables
% Loop through subjects and tasks
% for s = 1:length(sbj)
%     for k = 1:length(task)
%         %load the EEG data
%         % EEG = []; 
%         % Load the EEG data for each subject and task
%         [EEG, PATH] = o2_preprocessing(s, k, sbj, 20);
% 
%         % Select the stimulus of interest (as named in the EEG.event structure)
%         stim_label = 'alarm_tone_post';
% 
%         % Epoch the data with respect to the alarm tone
%         EEG_epoch = pop_epoch(EEG,{stim_label},[-1 2]) 
% 
%         % Save the data for the current subject and task
%         temp_dat(:,:,:,k) = EEG_epoch.data;
% 
%         % Define the filename for saving the preprocessed data
%         EEGFileName = sprintf('Preprocessed_EEG_Subject_%s_Task_%s.mat', sbj{s}, task{k});
% 
%         % Save the preprocessed EEG data for each subject and task
%         save(fullfile('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AllUsers\Preprocessed_Data\', EEGFileName), 'EEG');
% 
%     end
% end
%% Continue once you have a running pre-processing pipeline
% OT_setup

StatsParticipantTask = struct();

for s = 1:length(sbj)
    for k = 1:length(task)
        StatsParticipantTask(s, k).ParticipantID = s;  % s is the participant ID
        StatsParticipantTask(s, k).TaskID = k;         % k is the task ID
        % 
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task
        %%% Load the EEG and Audio
        [fs_eeg, resp, fs_audio, audio_dat] = LoadEEG(s, k, sbj, task);
        
        %%% FEATURE extraction 
        % Original Envelope Feature Generation   
        stim_Env = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox

        % Ensure stim_Env and resp have the same number of samples by cutting the larger one
        if size(stim_Env, 1) > size(resp, 1)
            % Cut stim_Env to match the size of resp
            stim_Env = stim_Env(1:size(resp, 1), :);
        elseif size(resp, 1) > size(stim_Env, 1)
            % Cut resp to match the size of stim_Env
            resp = resp(1:size(stim_Env, 1), :);
        end

        if ~isequal(size(stim_Env,1),size(resp,1))
            error(['STIM and RESP arguments must have the same number of '...
                'observations.'])
        end

        %%% ABenvelope Feature
        num_bins = 7;
        binEdges_dB = linspace(8, 64, num_bins + 1);        
        [~, stim_ABenv, NormEnv, NormBinEdges, BinEdges] = ABenvelopeGenerator_V2(stim_Env,binEdges_dB, num_bins);
        
        stim_ABenv_Norm = normalize(stim_ABenv,1,'range');

        %%% Onset Feature
        % stim_Env = stim_Env';                                   % Transpose the envelope
        % stim_Onset = OnsetGenerator(stim_Env); 
        % 
        % %%% Combine Top matrix of ABenvelope with Onset
        % CmbOnsPlusTopABenv = stim_ABenv; CmbOnsPlusTopABenv(:,1) = stim_Onset;  % ABEnvelope and Onset Concatinated (BinNum, all)
        %%% Amplitude Density
        % % Count amplitude density of each bin (Using histcounts)
        % [DbCounts, ~, ~] = histcounts(NormEnv, NormBinEdges');
        % % Plot Histogram
        % figure;
        % histogram('BinEdges',binEdges_dB,'BinCounts',DbCounts);
        % % Title, labels and flip, add grid
        % set(gca,'view',[90 -90])
        % xlabel('Bin Edges (dB)'); ylabel('Counts');
        % title('Histogram of Decibel Counts');
        % grid on;
        % Save figure
        % fig_filename = sprintf('Participant_%s_%s_Histogram', sbj{s},task{k});
        % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Histograms\',fig_filename)
        %%% Ready Features
        % stim_Env;           % stim_Env - Original Envelope (1, all)
        % stim_ABenv;           % stim_ABenv - Amplitude-Binned Envelope (BinNum, all)
        % CmbOnsPlusTopABenv;    % Cmb_Ons_ABenvTop_C - ABEnvelope and Onset Concatinated (BinNum, all)
        % feature_names = {'Orig Env (norm)', 'ABenv', 'CmbOnsPlusTopABenv'};
        % features = {NormEnv, stim_ABenv, CmbOnsPlusTopABenv};
%%
        %%% Using different features to get Accuracies
        feature_names = {'Orig Env (norm)', 'ABenv'};
        % features = {menv, env_bin_norm'};

        % Add the binned envelope to the features list
        features = {stim_Env, stim_ABenv_Norm};
        
        % Model hyperparameters
        tmin = -100;
        tmax = 400;
        % tmax = 500;
        trainfold = 10;
        testfold = 1;


        % Dir = 1; %specifies the forward modeling
        % 
        % lambdas = linspace(10e-4,10e4,10);
        % 
        % reg = cell(length(features), 1);
        % mod_w = cell(length(features), 1);

        figure; col_num = length(features); % Dinamic Figure Columns
            
        for feature_idx = 1:length(features)
            
            stim = features{feature_idx};  % Assign the current feature set to stim
            fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set
            % Partition the data into training and test data segments
            [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);
            
            % %%% z-score the input and output data
            % strainz = strain;
            % stestz = stest;
            % rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
            % rtestz = zscore(rtest, [], 'all');
            
            % %%% Use cross-validation to find the optimal regularization parameter
            % fs_eeg = EEG.srate;
            % cv = mTRFcrossval(strainz, rtrainz, fs, Dir, tmin, tmax, lambdas, 'Verbose', 0);

            % %get the optimal regression parameter
            l = mean(cv.r,3); %over channels
            [l_val,l_idx] = max(mean(l,1));
            l_opt = lambdas(l_idx);

            % Train the neural model with the optimal regularization parameter
            model = mTRFtrain(strainz, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);


            % Compute model weights
            model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);
            % Test the model on unseen neural data
            [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);
        
            % Plotting in a col_num x 2 grid
            
            % Model weights
            subplot(col_num, 2, feature_idx * 2 - 1)
            plot(model.t, squeeze(model.w(1, :, :))); % model weights

            %%%% (!)
            
            title(sprintf('Weights (%s)', feature_names{feature_idx}))
            ylabel('a.u.')
            xlabel('time in ms.')
            
            % GFP (Global Field Power)
            subplot(col_num, 2, feature_idx * 2)
            boxplot(stats.r) % channel correlation values
            title(sprintf('GFP (%s)', feature_names{feature_idx}))
            ylabel('correlation')

            % Store median correlation values
            % if feature_idx == 1                                     %NormEnv
            %     StatsParticipantTask(s, k).NormEnvMedCorr = median(stats.r);
            %     StatsParticipantTask(s, k).NormEnvModel = model;
            %     StatsParticipantTask(s, k).NormEnvStats = stats;
            % elseif feature_idx == 2                                 %ABenv
            %     StatsParticipantTask(s, k).ABenvMedCorr = median(stats.r);
            %     StatsParticipantTask(s, k).ABenvModel = model;
            %     StatsParticipantTask(s, k).ABenvStats = stats;
            % end
        
        end
        
        % Adjust the layout
        sgtitle('Feature Comparisons') % Add a main title to the figure
        
        % Save figure
        % fig_filename = sprintf('Participant_%s_Task_%s_Feature_%s', sbj{s}, task{k}, feature_names{feature_idx});
        % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Correlations\',fig_filename)
      
        % Run mTRFrun
        % [stats, model, strain, rtrain, stest, rtest] = mTRFrun(features, feature_names, resp, fs_eeg, tmin, tmax, testfold, trainfold);

        % clear stim_Env  stim_ABenv NormEnv NormBinEdges BinEdges num_bins fs_eeg resp fs_audio audio_dat;
        disp ('Results/Figures are saved!')
    end

end




%% V2 - Using different features to get Accuracies
% feature_names = {'Orig Env (norm)', 'ABenv', 'CmbOnsPlusTopABenv'};
feature_names = {'Orig Env (norm)', 'ABenv'};

% Add the binned envelope to the features list
% features = {NormEnv, stim_ABenv, CmbOnsPlusTopABenv};
features = {NormEnv, stim_ABenv};

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













%%
% % Save the StatsParticipantTask structure to a .mat file
% savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask.mat';
% save(savePath, 'StatsParticipantTask');     % Save the StatsParticipantTask structure to the specified path