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

%% Channel selection
[~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task);
eeg_channel_labels = {EEG.chanlocs.labels}; selected_channels = {'C3', 'FC2', 'FC1', 'Cz', 'C4'};
[~, channel_idx] = ismember(selected_channels, eeg_channel_labels);  % Find the indices corresponding to the selected channels in your EEG data
%% Search Grid
% Grid of bin ranges from 0 to 80 with different minimum values
bin_ranges = {    [0, 4, 4],    [0, 4, 8],    [0, 4, 16],    [0, 6, 3],    [0, 6, 6],    [0, 6, 9],    [0, 6, 12], [0, 8, 4],    [0, 8, 8],    [0, 8, 16],    [0, 10, 5],    [0, 10, 10],    [0, 10, 15],     [0, 20, 4],    [0, 20, 8],    [0, 20, 12],    [0, 40, 4],    [0, 40, 8],    [0, 40, 12],    [0, 60, 4],    [0, 60, 8],    [0, 60, 12],    [0, 80, 4],    [0, 80, 8],    [0, 80, 12],    [0, 80, 16],    [10, 20, 4],    [10, 20, 8],    [10, 20, 12],    [10, 40, 4],    [10, 40, 8],    [10, 40, 12],    [20, 40, 4],    [20, 40, 8],    [20, 40, 12],    [20, 60, 4],    [20, 60, 8],    [20, 60, 12],    [40, 80, 4],    [40, 80, 8],    [40, 80, 12]};

for z = 1:size(bin_ranges, 1)
    min_bin = bin_ranges{z}(1);
    max_bin = bin_ranges{z}(2);
    num_bins = bin_ranges{z}(3);
    
    % Generate bin edges
    binEdges_dB = linspace(min_bin, max_bin, num_bins + 1)';
    
    % (Optional) Display the bin edges
    fprintf('Min: %d, Max: %d, Num Bins: %d\n', min_bin, max_bin, num_bins);
    disp(binEdges_dB);



% % %% Continue once you have a running pre-processing pipeline
StatsParticipantTask_04_09 = struct();

for s = 1:length(sbj)
    for k = 1:length(task)

        % % % % %
        % s = 2;
        StatsParticipantTask_04_09(s, k).ParticipantID = s;  % s is the participant ID
        StatsParticipantTask_04_09(s, k).TaskID = k;         % k is the task ID
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task

        % Load the EEG and Audio
        [fs_eeg, full_resp, fs_audio, audio_dat, ~]  = LoadEEG(s, k, sbj, task);

        % Select only 5 channels from response
        resp = full_resp(:, channel_idx);

        % FEATURE extraction 

        % Original Envelope Feature Generation   
        Env = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox
        [Env, resp] = size_check(Env, resp); % Ensure matching sizes
        % NormEnv = normalize(stim_Env,1,'range');

        %%% ABenvelope Feature

 
        % % % % Parameters
        % min_bin = 0; num_bins = 8; max_bin = 8;
        % ABenvelope generation
        % [stim_ABenv_Norm, ~, NormEnv, NormBinEdges, ~, binEdges_dB]...
        %     = ABenvelopeGenerator_V2(stim_Env, min_bin, max_bin, num_bins);

        % Generate bin edges
        Linearity = 'Logarithmic';

        % if strcmp(Linearity, 'Logarithmic')
            % binEdges_dB = linspace(min_bin, max_bin, num_bins + 1)';
            binEdges_linear = 10.^(binEdges_dB / 20);
            NormBinEdges = normalize(binEdges_linear,1,'range');
        % elseif strcmp(Linearity, 'Linear')
        %         binEdges_dB = linspace(min_bin, max_bin, num_bins + 1)';
        %         NormBinEdges = normalize(binEdges_dB,1,'range');
        % end

        %%% Add Section for linear translation or without
    
        % Normalize
        EnvNorm = normalize(Env,1,'range');

        % Calculate the histogram counts and bin indices using histcounts
        % Count amplitude density of each bin (Using histcounts)
        [DbCounts, edges_1, binIndices] = histcounts(EnvNorm, NormBinEdges);

        %%% Histogram of Amplitude Density
        % figure; histogram('BinEdges',binEdges_dB','BinCounts',DbCounts); % Plot Histogram
        % % Title, labels and flip, add grid
        % set(gca,'view',[90 -90]); xlabel('Bin Edges (dB)'); 
        % ylabel('Counts'); title('Histogram of Decibel Counts'); grid on;

        % Initialize the binned envelope matrix
        ABenv = zeros(size(EnvNorm,1), num_bins);
        
        % Binned envelope
        for i = 1:length(binIndices)
            ABenv(i, binIndices(i)) = EnvNorm(i);
        end
        
        % Normalize each bin 
        ABenvNorm = normalize(ABenv, 1, 'range');
        
        % % % % Exclude bins 
        % % Calculate the percentage of values in each bin
        total_counts = sum(DbCounts); % Total number of values across all bins
        bin_percentages = (DbCounts / total_counts) * 100; % Percentage of values in each bin
        low_percentage_threshold = 5;           % Define a threshold for low percentage bins (e.g., less than 5%)
        low_percentage_bins = find(bin_percentages < low_percentage_threshold);        % Find the bins that have a percentage lower than the threshold

        %%%% Exclude the bins with low percentages
        ABenvRedBin = ABenv(:, low_percentage_bins(end)+1:end);
        ABenvNormRedBin = ABenvNorm(:, low_percentage_bins(end)+1:end);


        %(!)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

        %%% Histogram of Amplitude Density after exclusion
        % binEdges_dB_new = binEdges_dB(low_percentage_bins(end)+1:end);        % Remove these bins from the bin edges
        % NormBinEdges_new = NormBinEdges(low_percentage_bins(end)+1:end);        % Remove these bins from the bin edges
        % [DbCounts_new, edges_2, ~] = histcounts(ABenvNormRedBin, NormBinEdges_new); % size(stim_ABenv_Norm_new, 2)) % Get new bin counts        
        % figure; histogram('BinEdges',binEdges_dB_new','BinCounts',DbCounts_new); % Plot Histogram
        % % 
        % % Title, labels and flip, add grid
        % set(gca,'view',[90 -90]); xlabel('Bin Edges (dB)'); ylabel('Counts');
        % title('Histogram of Decibel Counts (After Exclusion)'); grid on;
        % Save figure
        % Hist_filename = sprintf('Participant_%s_%s_Histogram', sbj{s},task{k});
        % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Histograms\',Hist_filename)
        
        %(!)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%% Ready Features
        % % % % % Using different features to get Accuracies
        feature_names = {'EnvNorm', 'ABenv', 'ABenvNorm','ABenvRedBin','ABenvNormRedBin'};
        % Add the binned envelope to the features list
        features = {EnvNorm, ABenv, ABenvNorm, ABenvRedBin, ABenvNormRedBin};
        
        % Model hyperparameters
        tmin = -100;
        tmax = 400;         % tmax = 500;
        trainfold = 10;
        testfold = 1;
        Dir = 1;            %specifies the forward modeling

        lambdas = linspace(10e-4, 10e4,10);
        % figure; col_num = length(features); % Dinamic Figure Columns
            
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

            % Get the optimal regression parameter
            l = mean(cv.r,3); %over channels
            [l_val,l_idx] = max(mean(l,1));
            l_opt = lambdas(l_idx);

            % Compute model weights
            model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
            % Test the model on unseen neural data
            [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

            % Plotting in a col_num x 2 grid            
            % Model weights
            % subplot(col_num, 2, feature_idx * 2 - 1);
            % mean_model_w = squeeze(mean(model.w(:, :, :), 1));
            % plot(model.t, mean_model_w);  % plot(model.t, squeeze(model.w(5, :, :))); % model weights
            % title(sprintf('Weights (%s)', feature_names{feature_idx}));
            % ylabel('a.u.'); xlabel('time in ms.');

            % GFP (Global Field Power)
            % subplot(col_num, 2, feature_idx * 2);
            % boxplot(stats.r); ylim([-0.05 0.20]); % channel correlation values
            % title(sprintf('GFP (%s)', feature_names{feature_idx})); ylabel('correlation');

            % Store mean correlation values
            if feature_idx == 1                                     %NormEnv
                StatsParticipantTask_04_09(s, k).EnvNormModel = model;
                StatsParticipantTask_04_09(s, k).EnvNormStats = stats;
            elseif feature_idx == 2                                 %ABenv
                StatsParticipantTask_04_09(s, k).ABenvModel = model;
                StatsParticipantTask_04_09(s, k).ABenvStats = stats;
            elseif feature_idx == 3                                 %ABenv
                StatsParticipantTask_04_09(s, k).ABenvNormModel = model;
                StatsParticipantTask_04_09(s, k).ABenvNormStats = stats;
            elseif feature_idx == 4                                 %ABenv
                StatsParticipantTask_04_09(s, k).ABenvRedBinModel = model;
                StatsParticipantTask_04_09(s, k).ABenvRedBinStats = stats;
            elseif feature_idx == 5                                 %ABenv
                StatsParticipantTask_04_09(s, k).ABenvNormRedBinModel = model;
                StatsParticipantTask_04_09(s, k).ABenvNormRedBinStats = stats;
            end

        end
        
        % Adjust the layout
        % sgtitle(sprintf('Feature Comparisons: (%s)', task{k})) % Add a main title to the figure

        % Save figure
        % PearsonR_filename = sprintf('Participant_%s_Task_%s_Feature_%s', sbj{s}, task{k}, feature_names{feature_idx});
        % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Correlations\',PearsonR_filename)
      
        % Run mTRFrun
        % [stats, model, strain, rtrain, stest, rtest] = mTRFrun(features, feature_names, resp, fs_eeg, tmin, tmax, testfold, trainfold);

        % clear stim_Env  stim_ABenv NormEnv NormBinEdges BinEdges num_bins fs_eeg resp fs_audio audio_dat;
        disp ('Results/Figures are saved!')
    
    end

end

% %%

% % Save the StatsParticipantTask structure to a .mat file
% savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_new.mat';
% save(savePath, 'StatsParticipantTask_new');     % Save the StatsParticipantTask structure to the specified path


% %%
%%% plot that uses pearson r of normal envelope model and ab envelope over each subject
% clear
% OT_setup
% addpath("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data")
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_new.mat');

% %%
% [~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task);

% chanlocsData = EEG.chanlocs;
% figure; PlotTopoplot(StatsParticipantTask_new, chanlocsData, sbj);
% Topoplot_filename = sprintf('Topoplota');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Topoplot\',Topoplot_filename)

% %%
% Initialize arrays to store Pearson r values
% Narrow
narrow_EnvNormR_mean = NaN(length(sbj), 1);
narrow_ABenvR_mean = NaN(length(sbj), 1);
narrow_ABenvNormR_mean= NaN(length(sbj), 1);
narrow_ABenvRedBinR_mean= NaN(length(sbj), 1);
narrow_ABenvNormRedBinR_mean= NaN(length(sbj), 1);

% Wide
wide_EnvNormR_mean = NaN(length(sbj), 1);
wide_ABenvR_mean = NaN(length(sbj), 1);
wide_ABenvNormR_mean= NaN(length(sbj), 1);
wide_ABenvRedBinR_mean= NaN(length(sbj), 1);
wide_ABenvNormRedBinR_mean= NaN(length(sbj), 1);

% Extract Pearson r values for narrow and wide conditions
for s = 1:length(sbj)
    for k = 1:length(task)
        if isfield(StatsParticipantTask_04_09(s, k), 'EnvNormStats')
            if strcmp(task{k}, 'narrow') % Assuming task names are used to differentiate conditions
                narrow_EnvNormR_mean(s) = mean(StatsParticipantTask_04_09(s, k).EnvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_EnvNormR_mean(s) = mean(StatsParticipantTask_04_09(s, k).EnvNormStats.r);
            end
        end
        if isfield(StatsParticipantTask_04_09(s, k), 'ABenvStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvStats.r);
            end
        end
   	    if isfield(StatsParticipantTask_04_09(s, k), 'ABenvNormStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvNormR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvNormR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvNormStats.r);
            end
        end
        if isfield(StatsParticipantTask_04_09(s, k), 'ABenvRedBinStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvRedBinR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvRedBinStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvRedBinR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvRedBinStats.r);
            end
        end
        if isfield(StatsParticipantTask_04_09(s, k), 'ABenvNormRedBinStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvNormRedBinR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvNormRedBinStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvNormRedBinR_mean(s) = mean(StatsParticipantTask_04_09(s, k).ABenvNormRedBinStats.r);
            end

        end
    end
end


% Plot Pearson r values for narrow condition
figure;
hold on;
plot(1:length(sbj), narrow_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Narrow)');
plot(1:length(sbj), narrow_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Narrow)');
plot(1:length(sbj), narrow_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Narrow)');
plot(1:length(sbj), narrow_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins)');
plot(1:length(sbj), narrow_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins)');


xlabel('Subject');
ylabel('Pearson r (mean)');
title('Model NormEnv vs ABEnv - Narrow Condition');
legend('show');
ylim([-0.10 0.20])
grid on;
% Save the plot
% fig_filename_narrow = sprintf('PearsonsR_Narrow');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_narrow)


% Plot Pearson r values for wide condition
figure;
hold on;
plot(1:length(sbj), wide_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Wide)');
plot(1:length(sbj), wide_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Wide)');
plot(1:length(sbj), wide_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Wide)');
plot(1:length(sbj), wide_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins, Wide)');
plot(1:length(sbj), wide_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins, Wide)');
xlabel('Subject');
ylabel('Pearson r (mean)');
title('Model NormEnv vs ABEnv - Wide Condition');
legend('show');
grid on;
ylim([-0.10 0.20])
% Save the plot
% fig_filename_wide = sprintf('PearsonsR_Wide');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_wide)

end

%%
addpath('C:\Users\icbmadmin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Functions\Shapiro-Wilk and Shapiro-Francia normality tests')

% Shapiro-Wilk test for Narrow condition: NormEnv
[h_sw_NormEnv, p_sw_NormEnv, stats_sw_NormEnv] = swtest(narrow_EnvNormR_mean, 0.05);
if p_sw_NormEnv > 0.05
    disp(['Participant ' num2str(s) ', Condition: Narrow ; Shapiro-Wilk Test (NormEnv) p-value = ' num2str(p_sw_NormEnv) ' - Normally distributed']);
else
    disp(['Participant ' num2str(s) ', Condition: Narrow ; Shapiro-Wilk Test (NormEnv) p-value = ' num2str(p_sw_NormEnv) ' - Not normally distributed']);
end

% Shapiro-Wilk test for Narrow condition: ABEnv
[h_sw_ABEnv, p_sw_ABEnv, stats_sw_ABEnv] = swtest(narrow_ABenvR_mean, 0.05);
if p_sw_ABEnv > 0.05
    disp(['Participant ' num2str(s) ', Condition: Narrow ;  Shapiro-Wilk Test (ABEnv) p-value = ' num2str(p_sw_ABEnv) ' - Normally distributed']);
else
    disp(['Participant ' num2str(s) ', Condition: Narrow ;  Shapiro-Wilk Test (ABEnv) p-value = ' num2str(p_sw_ABEnv) ' - Not normally distributed']);
end

% Shapiro-Wilk test for Wide condition: NormEnv
[h_sw_WideNormEnv, p_sw_WideNormEnv, stats_sw_WideNormEnv] = swtest(wide_EnvNormR_mean, 0.05);
if p_sw_WideNormEnv > 0.05
    disp(['Participant ' num2str(s) ', Condition: Wide ;  Shapiro-Wilk Test (Wide NormEnv) p-value = ' num2str(p_sw_WideNormEnv) ' - Normally distributed']);
else
    disp(['Condition: Wide ;  Shapiro-Wilk Test (Wide NormEnv) p-value = ' num2str(p_sw_WideNormEnv) ' - Not normally distributed']);
end

% Shapiro-Wilk test for Wide condition: ABEnv
[h_sw_WideABEnv, p_sw_WideABEnv, stats_sw_WideABEnv] = swtest(wide_ABenvR_mean, 0.05);
if p_sw_WideABEnv > 0.05
    disp(['Condition: Wide ; Shapiro-Wilk Test (Wide ABEnv) p-value = ' num2str(p_sw_WideABEnv) ' - Normally distributed']);
else
    disp(['Condition: Wide ; Shapiro-Wilk Test (Wide ABEnv) p-value = ' num2str(p_sw_WideABEnv) ' - Not normally distributed']);
end

%%

% Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnv
[p_wilcoxon_Narrow, h_wilcoxon_Narrow] = signrank(narrow_EnvNormR_mean, narrow_ABenvR_mean, 'alpha', 0.05);
if p_wilcoxon_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - No significant difference']);
end

% Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
[p_wilcoxon_Wide, h_wilcoxon_Wide] = signrank(wide_EnvNormR_mean, wide_ABenvR_mean, 'alpha', 0.05);
if p_wilcoxon_Wide < 0.05
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - Significant difference']);
else
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - No significant difference']);
end
%%
% Create a figure for boxplots
figure;

% Narrow condition
subplot(1, 2, 1); % Create a subplot for Narrow condition
boxplot([narrow_EnvNormR_mean, narrow_ABenvR_mean], 'Labels', {'NormEnv', 'ABEnv'});
title('Narrow Condition: NormEnv vs ABEnv');
ylabel('Response Values');
grid on;

% Wide condition
subplot(1, 2, 2); % Create a subplot for Wide condition
boxplot([wide_EnvNormR_mean, wide_ABenvR_mean], 'Labels', {'NormEnv', 'ABEnv'});
title('Wide Condition: NormEnv vs ABEnv');
ylabel('Response Values');
grid on;


%%
figure;
% Narrow condition
subplot(1, 2, 1); % Create a subplot for Narrow condition
hold on;

% Plotting the first participant's data for Narrow Condition
plot(1, narrow_EnvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(1, :), 'DisplayName', 'NormEnv'); % NormEnv point
plot(2, narrow_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(2, :), 'DisplayName', 'ABEnv'); % ABEnv point

% Connect points with a line for clarity
plot([1, 2], [narrow_EnvNormR_mean, narrow_ABenvR_mean], 'k--'); % Dashed line connecting points

% Customize the plot
title('Narrow Condition: NormEnv vs ABEnv');
ylabel('Response Values');
xlim([0.5 2.5]); % Set X-axis limits
set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABEnv'}); % Set X-axis labels
grid on;


% Wide condition
subplot(1, 2, 2); % Create a subplot for Wide condition
hold on;

% Plotting the first participant's data for Wide Condition
plot(1, wide_EnvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(1, :), 'DisplayName', 'NormEnv'); % NormEnv point
plot(2, wide_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(2, :), 'DisplayName', 'ABEnv'); % ABEnv point

% Connect points with a line for clarity
plot([1, 2], [wide_EnvNormR_mean, wide_ABenvR_mean], 'k--'); % Dashed line connecting points

% Customize the plot
title('Wide Condition: NormEnv vs ABEnv');
ylabel('Response Values');
xlim([0.5 2.5]); % Set X-axis limits
set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABEnv'}); % Set X-axis labels
grid on;










%%
%%%%%%% Check how it works - Thorge's ABenv model
% menv_norm = stim_Env;
% binEdges_dB = 8:9:64;  % Binning in 9 dB steps up to 64 dB (adjust as needed)
% [env_bin_norm_T, env_bin_T, binEdges_normalized_T, binEdges_dB_T] = ABenvelopeGenerator_V2_Thorge_modified(menv_norm, binEdges_dB)
% features_T = {menv_norm, env_bin_norm_T};

%%%%%%% Onset Feature
% stim_Env = stim_Env';                                   % Transpose the envelope
% stim_Onset = OnsetGenerator(stim_Env); 
% 
% %%% Combine Top matrix of ABenvelope with Onset
% CmbOnsPlusTopABenv = stim_ABenv; CmbOnsPlusTopABenv(:,1) = stim_Onset;  % ABEnvelope and Onset Concatinated (BinNum, all)
