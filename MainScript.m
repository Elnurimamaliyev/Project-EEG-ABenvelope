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

%% Continue once you have a running pre-processing pipeline
StatsParticipantTask = struct();

for s = 1:length(sbj)
    for k = 1:length(task)
        StatsParticipantTask(s, k).ParticipantID = s;  % s is the participant ID
        StatsParticipantTask(s, k).TaskID = k;         % k is the task ID
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task

        %%% Load the EEG and Audio
        [fs_eeg, resp, fs_audio, audio_dat, EEG] = LoadEEG(s, k, sbj, task);
        
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
        num_bins = 8;
        binEdges_dB = linspace(30, 80, num_bins + 1);
        % binEdges_dB = linspace(8, 64, num_bins + 1);

        [stim_ABenv_Norm, stim_ABenv, NormEnv, NormBinEdges, BinEdges] = ABenvelopeGenerator_V2(stim_Env,binEdges_dB, num_bins);

        %%% Thorge's ABenv model
        % menv_norm = stim_Env;
        % binEdges_dB = 8:9:64;  % Binning in 9 dB steps up to 64 dB (adjust as needed)
        % [env_bin_norm_T, env_bin_T, binEdges_normalized_T, binEdges_dB_T] = ABenvelopeGenerator_V2_Thorge_modified(menv_norm, binEdges_dB)
        % features_T = {menv_norm, env_bin_norm_T};

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

        %%% Using different features to get Accuracies
        feature_names = {'Orig Env (norm)', 'ABenv (norm)'};

        % Add the binned envelope to the features list
        features = {stim_Env, stim_ABenv_Norm};

        % Model hyperparameters
        tmin = -100;
        tmax = 400;
        % tmax = 500;
        trainfold = 10;
        testfold = 1;


        Dir = 1; %specifies the forward modeling
        lambdas = linspace(10e-4, 10e4,10);

        figure; col_num = length(features); % Dinamic Figure Columns
            
        for feature_idx = 1:length(features)
            
            stim = features{feature_idx};  % Assign the current feature set to stim
            fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set
            % Partition the data into training and test data segments
            [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);
            
            %%% z-score the input and output data
            rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
            rtestz = zscore(rtest, [], 'all');
            
            %%% Use cross-validation to find the optimal regularization parameter
            cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);

            %get the optimal regression parameter
            l = mean(cv.r,3); %over channels
            [l_val,l_idx] = max(mean(l,1));
            l_opt = lambdas(l_idx);

            % Compute model weights
            model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
            % Test the model on unseen neural data
            [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

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
            if feature_idx == 1                                     %NormEnv
                StatsParticipantTask(s, k).NormEnvMedCorr = median(stats.r);
                StatsParticipantTask(s, k).NormEnvModel = model;
                StatsParticipantTask(s, k).NormEnvStats = stats;
            elseif feature_idx == 2                                 %ABenv
                StatsParticipantTask(s, k).ABenvMedCorr = median(stats.r);
                StatsParticipantTask(s, k).ABenvModel = model;
                StatsParticipantTask(s, k).ABenvStats = stats;
            end

        end
        
        % Adjust the layout
        sgtitle('Feature Comparisons') % Add a main title to the figure

        % Save figure
        fig_filename = sprintf('Participant_%s_Task_%s_Feature_%s', sbj{s}, task{k}, feature_names{feature_idx});
        % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Correlations\',fig_filename)
      
        % Run mTRFrun
        % [stats, model, strain, rtrain, stest, rtest] = mTRFrun(features, feature_names, resp, fs_eeg, tmin, tmax, testfold, trainfold);

        % clear stim_Env  stim_ABenv NormEnv NormBinEdges BinEdges num_bins fs_eeg resp fs_audio audio_dat;
        disp ('Results/Figures are saved!')
    
    end

end


% % Save the StatsParticipantTask structure to a .mat file
savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask.mat';
save(savePath, 'StatsParticipantTask');     % Save the StatsParticipantTask structure to the specified path


%%
%%% plot that uses pearson r of normal envelope model and ab envelope over each subject
clear
OT_setup
addpath("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data")

load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask.mat');





%%

chanlocsData = EEG.chanlocs;
PlotTopoplot(StatsParticipantTask, chanlocsData, sbj)

%%
% Initialize arrays to store Pearson r values
narrow_norm_env_r_median = NaN(length(sbj), 1);
narrow_ab_env_r_median = NaN(length(sbj), 1);
wide_norm_env_r_median = NaN(length(sbj), 1);
wide_ab_env_r_median = NaN(length(sbj), 1);

% Extract Pearson r values for narrow and wide conditions
for s = 1:length(sbj)
    for k = 1:length(task)
        if isfield(StatsParticipantTask(s, k), 'NormEnvMedCorr')
            if strcmp(task{k}, 'narrow') % Assuming task names are used to differentiate conditions
                narrow_norm_env_r_median(s) = StatsParticipantTask(s, k).NormEnvMedCorr;
            elseif strcmp(task{k}, 'wide')
                wide_norm_env_r_median(s) = StatsParticipantTask(s, k).NormEnvMedCorr;
            end
        end
        if isfield(StatsParticipantTask(s, k), 'ABenvMedCorr')
            if strcmp(task{k}, 'narrow')
                narrow_ab_env_r_median(s) = StatsParticipantTask(s, k).ABenvMedCorr;
            elseif strcmp(task{k}, 'wide')
                wide_ab_env_r_median(s) = StatsParticipantTask(s, k).ABenvMedCorr;
            end
        end
    end
end


%%
% Plot Pearson r values for narrow condition
figure;
hold on;
plot(1:length(sbj), narrow_norm_env_r_median, 'bo-', 'DisplayName', 'Normal Envelope (Narrow)');
plot(1:length(sbj), narrow_ab_env_r_median, 'ro-', 'DisplayName', 'AB Envelope (Narrow)');
xlabel('Subject');
ylabel('Pearson r (Median)');
title('Model NormEnv vs ABEnv - Narrow Condition');
legend('show');
ylim([-0.15 0.16])
grid on;
% Save the plot
fig_filename_narrow = sprintf('PearsonsR_Narrow');
save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_narrow)


% Plot Pearson r values for wide condition
figure;
hold on;
plot(1:length(sbj), wide_norm_env_r_median, 'bo-', 'DisplayName', 'Normal Envelope (Wide)');
plot(1:length(sbj), wide_ab_env_r_median, 'ro-', 'DisplayName', 'AB Envelope (Wide)');
xlabel('Subject');
ylabel('Pearson r (Median)');
title('Model NormEnv vs ABEnv - Wide Condition');
legend('show');
grid on;
ylim([-0.16 0.21])
% Save the plot
fig_filename_wide = sprintf('PearsonsR_Wide');
save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_wide)