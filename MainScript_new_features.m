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
clear; clc;

% Add Main paths
OT_setup
%% Channel selection
[~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task);
eeg_channel_labels = {EEG.chanlocs.labels}; selected_channels = {'C3', 'FC2', 'FC1', 'Cz', 'C4'};
[~, channel_idx] = ismember(selected_channels, eeg_channel_labels);  % Find the indices corresponding to the selected channels in your EEG data
%% Continue once you have a running pre-processing pipeline
StatsParticipantTask_28_10 = struct();
for s = 1:length(sbj)
    for k = 1:length(task)
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task
        % Load the EEG and Audio
        [fs_eeg, full_resp, fs_audio, audio_dat, ~]  = LoadEEG(s, k, sbj, task);

        % Select only 5 channels from response
        resp = full_resp(:, channel_idx);
        % resp = full_resp;

        [EnvNorm, ABenvNorm, Onset, OnsetEnvelope, resp]...
        = feature_deriv(audio_dat, fs_audio, fs_eeg, resp);

        OnsetandEnvelope = [ABenvNorm(:,1:8), Onset];
        norm_onsetEnvelope = normalize(OnsetEnvelope,1,'range');

        ABenvOnsetenv = [ABenvNorm(:,1:9), norm_onsetEnvelope];


        [EnvNorm, resp] = size_check(EnvNorm, resp);

        % Feature list
        % feature_names = {'EnvNorm', 'ABenvNorm', 'ABenvOnsetenv',  'OnsetandEnvelope'};
        % features = {EnvNorm, ABenvNorm, OnsetEnvelope, ABenvOnsetenv, OnsetandEnvelope};
        feature_names = {'EnvNorm', 'ABenvNorm', 'ABenvOnsetenv', 'OnsetEnvelope'};
        features = {EnvNorm, ABenvNorm, ABenvOnsetenv, OnsetEnvelope};
        % Model hyperparameters
        tmin = -100;
        tmax = 400;         % tmax = 500;
        trainfold = 10;
        testfold = 1;
        Dir = 1;            % specifies the forward modeling

        lambdas = linspace(10e-4, 10e4,10);
        for feature_idx = 1:length(features)
            
            stim = features{feature_idx};  % Assign the current feature set to stim

            size_check(stim, resp);  % Ensure matching sizes


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

            % Store mean correlation values
            if feature_idx == 1                                     %NormEnv
                StatsParticipantTask_28_10(s, k).standardEnvelopeModel = model;
                StatsParticipantTask_28_10(s, k).standardEnvelopeStats = stats;
            elseif feature_idx == 2                                 %ABons
                StatsParticipantTask_28_10(s, k).ABenvNormModel = model;
                StatsParticipantTask_28_10(s, k).ABenvNormStats = stats;
            elseif feature_idx == 3                                 %ABenv
                StatsParticipantTask_28_10(s, k).ABenvOnsetenvModel = model;
                StatsParticipantTask_28_10(s, k).ABenvOnsetenvStats = stats;
            elseif feature_idx == 4                                 %Onset
                    StatsParticipantTask_28_10(s, k).OnsetEnvelopeModel = model;
                    StatsParticipantTask_28_10(s, k).OnsetEnvelopeStats = stats;
            end
                % elseif feature_idx == 4                                 %Onset
            %         StatsParticipantTask_28_10(s, k).OnsetandEnvelopeModel = model;
            %         StatsParticipantTask_28_10(s, k).OnsetandEnvelopeStats = stats;
            % end
        end
        % Save Participant and Task IDs
        StatsParticipantTask_28_10(s, k).ParticipantID = s;  % s is the participant ID
        StatsParticipantTask_28_10(s, k).TaskID = k;         % k is the task ID

        % clear stim_Env  stim_ABenv NormEnv NormBinEdges BinEdges num_bins fs_eeg resp fs_audio audio_dat;
        disp ('Results are saved!')
    
    end
end


%% Topoplot
%%% plot that uses pearson r of normal envelope model and ab envelope over each subject
% clear
% OT_setup
% addpath("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data")
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_new.mat');
% %%
% [~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task);
% 
% chanlocsData = EEG.chanlocs;
% figure; PlotTopoplot(StatsParticipantTask_04_09, chanlocsData, sbj);
% Topoplot_filename = sprintf('Topoplota');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Topoplot\',Topoplot_filename)

%%
OT_setup
% name_struct_new = StatsParticipantTask_0_096_10;
name_struct_new = StatsParticipantTask_28_10;

%%
% Initialize arrays to store Pearson r values
% Narrow

narrow_standardEnvR_mean = NaN(length(sbj), 1);
narrow_ABenvOnsetR_mean = NaN(length(sbj), 1);
% narrow_OnsetandEnvelopeR_mean = NaN(length(sbj), 1);
narrow_ABenvNormR_mean = NaN(length(sbj), 1);
narrow_OnsetEnvR_mean = NaN(length(sbj), 1);
% Wide

wide_standardEnvR_mean = NaN(length(sbj), 1);
wide_ABenvOnsetR_mean = NaN(length(sbj), 1);
% wide_OnsetandEnvelopeR_mean = NaN(length(sbj), 1);
wide_ABenvNormR_mean = NaN(length(sbj), 1);
wide_OnsetEnvR_mean = NaN(length(sbj), 1);


% DbCounts_all = cell(length(sbj), length(task)); 
% binEdges_dB_all = cell(length(sbj), length(task));

% Extract Pearson r values for narrow and wide conditions
for s = 1:length(sbj)
    for k = 1:length(task)
        

        if isfield(name_struct_new(s, k), 'standardEnvelopeStats')
            if strcmp(task{k}, 'narrow') % Assuming task names are used to differentiate conditions
                narrow_standardEnvR_mean(s) = mean(name_struct_new(s, k).standardEnvelopeStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_standardEnvR_mean(s) = mean(name_struct_new(s, k).standardEnvelopeStats.r);
            end
        end
        if isfield(name_struct_new(s, k), 'ABenvNormStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvNormR_mean(s) = mean(name_struct_new(s, k).ABenvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvNormR_mean(s) = mean(name_struct_new(s, k).ABenvNormStats.r);
            end
        end
        if isfield(name_struct_new(s, k), 'ABenvOnsetenvStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvOnsetR_mean(s) = mean(name_struct_new(s, k).ABenvOnsetenvStats.r);
            elseif strcmp(task{k}, 'wide')  
                wide_ABenvOnsetR_mean(s) = mean(name_struct_new(s, k).ABenvOnsetenvStats.r);
            end
        end
        % if isfield(name_struct_new(s, k), 'OnsetandEnvelopeStats')
        %     if strcmp(task{k}, 'narrow')
        %         narrow_OnsetandEnvelopeR_mean(s) = mean(name_struct_new(s, k).OnsetandEnvelopeStats.r);
        %     elseif strcmp(task{k}, 'wide')
        %         wide_OnsetandEnvelopeR_mean(s) = mean(name_struct_new(s, k).OnsetandEnvelopeStats.r);
        %     end
        % end
        if isfield(name_struct_new(s, k), 'OnsetEnvelopeStats')
            if strcmp(task{k}, 'narrow')
                narrow_OnsetEnvR_mean(s) = mean(name_struct_new(s, k).OnsetEnvelopeStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_OnsetEnvR_mean(s) = mean(name_struct_new(s, k).OnsetEnvelopeStats.r);
            end
        end

    end
end



%% Combination
% meanstandardEnvR=(narrow_standardEnvR_mean + wide_standardEnvR_mean)/2;
meanABEnvR=(narrow_ABenvNormR_mean + wide_ABenvNormR_mean)/2;
meancombinedR=(narrow_ABenvOnsetR_mean+wide_ABenvOnsetR_mean)/2;
meanOnsetEnvR=(narrow_OnsetEnvR_mean+wide_OnsetEnvR_mean)/2;

[sorted_meanABEnvR_mean, sortIdxcombinedmean] = sort(meanABEnvR, 'ascend');
sorted_meanCombinedR_mean = meancombinedR(sortIdxcombinedmean);
sorted_meanOnsetEnvR_mean = meanOnsetEnvR(sortIdxcombinedmean);

figure;
plot(1:length(sbj), sorted_meanABEnvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'ABenv', 'LineWidth', 2, 'MarkerSize', 5);

hold on
plot(1:length(sbj), sorted_meanOnsetEnvR_mean, 'o-', 'Color', [1, 0.75, 0.25], 'DisplayName', 'Onset Envelope', 'LineWidth', 2, 'MarkerSize', 5);
plot(1:length(sbj), sorted_meanCombinedR_mean, 'o-', 'Color', [0.1 0.5 0.1], 'DisplayName', 'Combined (ABenv+Onset Env)', 'LineWidth', 2, 'MarkerSize', 5);

xlabel('Subjects (sorted by Standard Env values)', 'FontWeight', 'bold', 'FontSize', 12); 
ylabel("Pearson's r (mean)");
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);
xlim([0.5 20.5]);
title('Mean of both conditions results', 'FontWeight', 'bold'); 
grid on; hold on; box off;
yline(0, '--k', 'DisplayName', 'near zero indicator', 'LineWidth', 1.5);

% yline(chance_level, '--k', 'DisplayName', 'Chance Level (+0.005)', 'LineWidth', 1.5);
% yline(-chance_level, '--k', 'DisplayName', 'Chance Level (-0.005)', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 
set(gca, 'FontSize', 13);
%%
Box_and_Ranked_Plot(meanstandardEnvR, meanABEnvR, wide_standardEnvR_mean, wide_ABenvNormR_mean)


%%
meanABEnvR=(narrow_ABenvNormR_mean + wide_ABenvNormR_mean)/2;
meancombinedR=(narrow_ABenvOnsetR_mean+wide_ABenvOnsetR_mean)/2;
meanOnsetEnvR=(narrow_OnsetEnvR_mean+wide_OnsetEnvR_mean)/2;

[h_ttest_Mean, p_ttest_Mean, ci_Narrow, stats_Narrow] = ttest(meanABEnvR, meancombinedR);
d_Narrow = stats_Narrow.tstat / sqrt(length(meanABEnvR)); % Cohen's d

if p_ttest_Mean < 0.05
    disp(['Condition: Average - (narrow+wide)/2 ; Paired t-Test (ABEnv vs Combined) p-value = ' num2str(p_ttest_Mean) ' - Significant difference, Cohen''s d = ' num2str(d_Narrow)]);
else
    disp(['Condition: Average - (narrow+wide)/2 ; Paired t-Test (ABEnv vs Combined) p-value = ' num2str(p_ttest_Mean) ' - No significant difference']);
end

% Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
[p_wilcoxon_Mean, h_wilcoxon_Mean] = signrank(meanABEnvR, meancombinedR, 'alpha', 0.05);
if p_wilcoxon_Mean < 0.05
    disp(['Condition: Average - (narrow+wide)/2 ; Wilcoxon signed-rank Test (ABEnv vs Combined) p-value = ' num2str(p_wilcoxon_Mean) ' - Significant difference']);
else
    disp(['Condition: Average - (narrow+wide)/2 ; Wilcoxon signed-rank Test (ABEnv vs Combined) p-value = ' num2str(p_wilcoxon_Mean) ' - No significant difference']);
end


%% Plot Pearson r values
% %%
% Sort increasing order based on the updated standard envelope variables
[sorted_narrow_standardEnvR_mean, sortIdx_narrow] = sort(narrow_standardEnvR_mean, 'ascend');

% Apply the same sorting for all narrow condition envelopes
sorted_narrow_ABenvR_mean = narrow_ABenvNormR_mean(sortIdx_narrow);
sorted_narrow_OnsetandEnvelopeR_mean = narrow_OnsetandEnvelopeR_mean(sortIdx_narrow);
sorted_narrowABenvOnsetStatsR_mean = narrow_ABenvOnsetR_mean(sortIdx_narrow);

[sorted_wide_standardEnvR_mean, sortIdx_wide] = sort(wide_standardEnvR_mean, 'ascend');

% Apply the same sorting for all wide condition envelopes
sorted_wide_ABenvR_mean = wide_ABenvNormR_mean(sortIdx_wide);
sorted_wide_OnsetandEnvelopeR_mean = wide_OnsetandEnvelopeR_mean(sortIdx_wide);
sorted_wideABenvOnsetStatsR_mean = wide_ABenvOnsetR_mean(sortIdx_wide);

figure;

% Narrow Condition
subplot(1,2,1); hold on;
% plot(1:length(sbj), sorted_narrow_standardEnvR_mean, 'o-', 'DisplayName', 'Standard Env', 'LineWidth', 2, 'MarkerSize', 5);
plot(1:length(sbj), sorted_narrow_ABenvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'AB Env', 'LineWidth', 2, 'MarkerSize', 5);
plot(1:length(sbj), sorted_narrow_OnsetandEnvelopeR_mean, 'o-', 'Color', [0.1 0.5 0.1], 'DisplayName', 'OnsetandEnvelope', 'LineWidth', 2, 'MarkerSize', 5); % Onset Envelope
plot(1:length(sbj), sorted_narrowABenvOnsetStatsR_mean, 'o-', 'Color', [0.8 0.3 0], 'DisplayName', 'ABenvOnset', 'LineWidth', 2, 'MarkerSize', 5);
xlabel('Subjects (sorted by Standard Env values)', 'FontWeight', 'bold'); 
ylabel("Pearson's r (mean)");
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);
xlim([0.5 20.5]);
title('Narrow Condition', 'FontWeight', 'bold');
grid on; hold on;
yline(0, '--k', 'DisplayName', 'Near zero indicator', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 
set(gca, 'FontSize', 13);

% Wide Condition
subplot(1,2,2); hold on;
% plot(1:length(sbj), sorted_wide_standardEnvR_mean, 'o-', 'DisplayName', 'Standard Env', 'LineWidth', 2, 'MarkerSize', 5);
plot(1:length(sbj), sorted_wide_ABenvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'AB Env', 'LineWidth', 2, 'MarkerSize', 5);
plot(1:length(sbj), sorted_wide_OnsetandEnvelopeR_mean, 'o-', 'Color', [0.1 0.5 0.1], 'DisplayName', 'OnsetandEnvelope', 'LineWidth', 2, 'MarkerSize', 5); % Onset Envelope
plot(1:length(sbj), sorted_wideABenvOnsetStatsR_mean, 'o-', 'Color', [0.8 0.3 0], 'DisplayName', 'ABenvOnset', 'LineWidth', 2, 'MarkerSize', 5); 
xlabel('Subjects (sorted by Standard Env values)', 'FontWeight', 'bold', 'FontSize', 12); 
ylabel("Pearson's r (mean)");
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);
xlim([0.5 20.5]);
title('Wide Condition', 'FontWeight', 'bold'); 
grid on; hold on;
yline(0, '--k', 'DisplayName', 'Near zero indicator', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 
set(gca, 'FontSize', 13);

% Save the plot
% fig_filename = sprintf('PearsonsR_max_itself');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename)
%% Box plot         narrow_ABenvNormR_mean         narrow_OnsetandEnvelopeR_mean       narrow_standardEnvR_mean        narrowABenvOnsetStatsR_mean
Box_and_Ranked_Plot(narrow_ABenvNormR_mean, narrow_ABenvOnsetR_mean, wide_standardEnvR_mean, wide_ABenvNormR_mean)
%%
Box_and_Ranked_Plot(meanstandardEnvR, meanABEnvR, wide_standardEnvR_mean, wide_ABenvNormR_mean)

%% 
meanstandardEnvR=(narrow_standardEnvR_mean + wide_standardEnvR_mean)/2;
meanABEnvR=(narrow_ABenvNormR_mean + wide_ABenvNormR_mean)/2;

[sorted_meanstandardEnvR_mean, sortIdxmean] = sort(meanstandardEnvR, 'ascend');
sorted_meanABEnvR_mean = meanABEnvR(sortIdxmean);
figure;
plot(1:length(sbj), sorted_meanstandardEnvR_mean, 'o-', 'DisplayName', 'Standard Env', 'LineWidth', 2, 'MarkerSize', 5);
hold on
plot(1:length(sbj), sorted_meanABEnvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'AB Env', 'LineWidth', 2, 'MarkerSize', 5);

xlabel('Subjects (sorted by Standard Env values)', 'FontWeight', 'bold', 'FontSize', 12); 
ylabel("Pearson's r (mean)");
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);
xlim([0.5 20.5]);
title('Mean of both conditions results', 'FontWeight', 'bold'); 
grid on; hold on; box off;
yline(0, '--k', 'DisplayName', 'near zero indicator', 'LineWidth', 1.5);

% yline(chance_level, '--k', 'DisplayName', 'Chance Level (+0.005)', 'LineWidth', 1.5);
% yline(-chance_level, '--k', 'DisplayName', 'Chance Level (-0.005)', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 
set(gca, 'FontSize', 13);

%%
% Box_and_Ranked_Plot(narrow_standardEnvR_mean, narrow_OnsetandEnvelopeR_mean, wide_standardEnvR_mean, wide_OnsetandEnvelopeR_mean)
% Box_and_Ranked_Plot(narrow_ABenvNormR_mean, narrow_OnsetandEnvelopeR_mean, wide_ABenvNormR_mean, wide_OnsetandEnvelopeR_mean)

%% Histogram

binedges_var = binEdges_dB_all{1};

for k = 1:length(task)
    figure;
    for s = 1:length(sbj)

    subplot(length(sbj)/5, 5, s)
    histogram('BinEdges',edges','BinCounts',DbCounts_all{s,k}); % Plot Histogram
    % Title, labels and flip, add grid
    set(gca,'view',[90 -90]); xlabel('Bin Edges (dB)'); 
    title(sprintf('Subject (%s) Decibel Counts', sbj{s})); grid on; 
    end
end

DbCounts_all_narrow = zeros(20, 8);
DbCounts_all_wide = zeros(20, 8);

for s = 1:length(sbj)
        % Extract DbCounts and binEdges_dB for the current subject and task
        DbCounts_all_narrow(s,:) = StatsParticipantTask_28_10(s, 1).DbCounts;
        DbCounts_all_wide(s,:) = StatsParticipantTask_28_10(s, 2).DbCounts;
end

mean_DbCounts_all_narrow = mean(DbCounts_all_narrow, 1);
mean_DbCounts_all_wide = mean(DbCounts_all_wide, 1);

figure;
% First subplot for narrow histogram
subplot(1, 2, 1);
histogram('BinEdges', binEdges_dB', 'BinCounts', mean_DbCounts_all_narrow); % Plot Histogram
% Title, labels and flip
xlabel('Amplitude Bins (Normalized)'); ylabel('Counts'); 
title('Mean of the Subject Decibel Counts (Narrow)'); 
box off; xtickformat('%.3f'); xticks(binEdges_dB); ytickformat('%.3f');
set(gca,'view',[90 -90], 'YTickLabel', num2str(yticks', '%.0f'));

% Second subplot for wide histogram
subplot(1, 2, 2);
histogram('BinEdges', binEdges_dB', 'BinCounts', mean_DbCounts_all_wide); % Plot Histogram
% Title, labels and flip
xlabel('Amplitude Bins (Normalized)'); ylabel('Counts'); 
title('Mean of the Subject Decibel Counts (Wide)'); 
box off; xtickformat('%.3f'); xticks(binEdges_dB); ytickformat('%.3f');
set(gca,'view',[90 -90], 'YTickLabel', num2str(yticks', '%.0f'));



%%

%%

    %%% Histogram of Amplitude Density
    % figure; histogram('BinEdges',binEdges_dB','BinCounts',DbCounts); % Plot Histogram
    % % Title, labels and flip, add grid
    % set(gca,'view',[90 -90]); xlabel('Bin Edges (dB)'); 
    % ylabel('Counts'); title('Histogram of Decibel Counts'); grid on;

    % figure; col_num = length(features); % Dinamic Figure Columns
    
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
    
    % % GFP (Global Field Power)
    % subplot(1,col_num, feature_idx);
    % boxplot(stats.r); ylim([-0.05 0.20]); % channel correlation values
    % title(sprintf('GFP (%s)', feature_names{feature_idx})); ylabel('correlation');


    % Adjust the layout
    % sgtitle(sprintf('Feature Comparisons: (%s)', task{k})) % Add a main title to the figure

    % Save figure
    % PearsonR_filename = sprintf('Participant_%s_Task_%s_Feature_%s', sbj{s}, task{k}, feature_names{feature_idx});
    % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Correlations\',PearsonR_filename)
  
    % Run mTRFrun
    % [stats, model, strain, rtrain, stest, rtest] = mTRFrun(features, feature_names, resp, fs_eeg, tmin, tmax, testfold, trainfold);
%% Normality test - Shapiro-Wilk test
% Add the function
addpath('C:\Users\icbmadmin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Functions\Shapiro-Wilk and Shapiro-Francia normality tests')
% Name the variables
variables = {narrow_EnvNormR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvNormR_mean};
% Run for loop over variables to check the normality
for i = 1:length(variables)
    [h_sw_NormEnv, p_sw_NormEnv, stats_sw_NormEnv] = swtest(variables{i}, 0.05);
    if p_sw_NormEnv > 0.05
        disp(['Shapiro-Wilk Test, p-value = ' num2str(p_sw_NormEnv) ' - Normally distributed']);
    else
        disp(['Shapiro-Wilk Test, p-value = ' num2str(p_sw_NormEnv) ' - Not normally distributed']);
    end
end
%%

% Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnv
[p_wilcoxon_Narrow, h_wilcoxon_Narrow] = signrank(narrow_ABenvNormR_mean, narrow_OnsetandEnvelopeR_mean, 'alpha', 0.05);
if p_wilcoxon_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - No significant difference']);
end

% Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnv
[p_wilcoxon_Narrow, h_wilcoxon_Narrow] = signrank(narrow_ABenvNormR_mean, narrow_ABenvOnsetR_mean, 'alpha', 0.05);
if p_wilcoxon_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - No significant difference']);
end
%%
% Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnv
[p_wilcoxon_Narrow, h_wilcoxon_Narrow] = signrank(narrow_standardEnvR_mean, narrow_ABenvOnsetR_mean, 'alpha', 0.05);
if p_wilcoxon_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Narrow) ' - No significant difference']);
end

%%
% Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
[p_wilcoxon_Wide, h_wilcoxon_Wide] = signrank(wide_standardEnvR_mean, wide_ABenvNormR_mean, 'alpha', 0.05);
if p_wilcoxon_Wide < 0.05
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - Significant difference']);
else
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - No significant difference']);
end

% Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
[p_wilcoxon_Wide, h_wilcoxon_Wide] = signrank(wide_standardEnvR_mean, wide_OnsetandEnvelopeR_mean, 'alpha', 0.05);
if p_wilcoxon_Wide < 0.05
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - Significant difference']);
else
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_wilcoxon_Wide) ' - No significant difference']);
end

%%
meanstandardEnvR=(narrow_standardEnvR_mean + wide_standardEnvR_mean)/2;
meanABEnvR=(narrow_ABenvNormR_mean + wide_ABenvNormR_mean)/2;

[h_ttest_Mean, p_ttest_Mean, ci_Narrow, stats_Narrow] = ttest(meanstandardEnvR, meanABEnvR);
d_Narrow = stats_Narrow.tstat / sqrt(length(meanstandardEnvR)); % Cohen's d

if p_ttest_Mean < 0.05
    disp(['Condition: Average - (narrow+wide)/2 ; Paired t-Test (Env vs ABEnv) p-value = ' num2str(p_ttest_Mean) ' - Significant difference, Cohen''s d = ' num2str(d_Narrow)]);
else
    disp(['Condition: Average - (narrow+wide)/2 ; Paired t-Test (Env vs ABEnv) p-value = ' num2str(p_ttest_Mean) ' - No significant difference']);
end

% Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
[p_wilcoxon_Mean, h_wilcoxon_Mean] = signrank(meanstandardEnvR, meanABEnvR, 'alpha', 0.05);
if p_wilcoxon_Mean < 0.05
    disp(['Condition: Average - (narrow+wide)/2 ; Wilcoxon signed-rank Test (Env vs ABEnv) p-value = ' num2str(p_wilcoxon_Mean) ' - Significant difference']);
else
    disp(['Condition: Average - (narrow+wide)/2 ; Wilcoxon signed-rank Test (Env vs ABEnv) p-value = ' num2str(p_wilcoxon_Mean) ' - No significant difference']);
end
%% 

    % Define colors for the different conditions
    color_NormEnv = [0, 0.4470, 0.7410]; % Blue
    color_ABenv = [0.5 0 0.5]; % purple

    color_Lines = [0.2, 0.2, 0.2]; % Grey for connecting lines
    
    % Create a figure with better proportions
    figure; 
   
    % Narrow Condition Plot (Left)
    hold on;

    % Boxplot for Narrow Condition
    h = boxplot([meanstandardEnvR, meanABEnvR], ...
                'Labels', {'NormEnv', 'ABenv'}, 'Widths', 0.5);
    
    % Set properties for boxplot appearance
    set(h, 'LineWidth', 2); % Thicker lines for boxplot
    
    % Overlay points and lines for individual participants
    plot(1, meanstandardEnvR, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_NormEnv, ...
         'MarkerFaceColor', color_NormEnv, 'DisplayName', 'NormEnv'); % NormEnv points
    plot(2, meanABEnvR, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_ABenv, ...
         'MarkerFaceColor', color_ABenv, 'DisplayName', 'ABenv'); % ABEnv points
     
    % Connect corresponding points with lines
    for i = 1:length(meanstandardEnvR)
        plot([1, 2], [meanstandardEnvR(i), meanABEnvR(i)], 'Color', color_Lines, 'LineWidth', 1.2);
    end
    
    % Customize the plot
    title('Average: (narrow+wide)/2: NormEnv vs ABenv', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Response Values', 'FontSize', 12);
    xlim([0.5 2.5]); % Set X-axis limits
    ylim([-0.03 0.19]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABenv'}, 'FontSize', 10);
    grid on;    box off;

    % Tighten the layout and improve spacing
    set(gcf, 'Color', 'w'); % Set background color to white
    set(gca, 'FontSize', 14);


%%
% Paired T test for Narrow condition: NormEnv vs ABEnv
[h_ttest_Narrow, p_ttest_Narrow] = ttest(narrow_standardEnvR_mean, narrow_ABenvNormR_mean);
if p_ttest_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Narrow) ' - No significant difference']);
end

% Paired T test for Wide condition: NormEnv vs ABEnv
[h_ttest_Wide,p_ttest_Wide] = ttest(wide_standardEnvR_mean, wide_ABenvNormR_mean);
if p_ttest_Wide < 0.05
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Wide) ' - Significant difference']);
else
    disp(['Condition: Wide ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Wide) ' - No significant difference']);
end
%%
% Create a figure for boxplots
figure;

% Narrow condition
subplot(1, 2, 1); % Create a subplot for Narrow condition
boxplot([narrow_EnvNormR_mean, narrow_ABenvNormR_mean], 'Labels', {'NormEnv', 'ABEnv'});
title('Narrow Condition: NormEnv vs ABEnv');
ylabel('Response Values');
grid on;

% Wide condition
subplot(1, 2, 2); % Create a subplot for Wide condition
boxplot([wide_EnvNormR_mean, wide_ABenvNormR_mean], 'Labels', {'NormEnv', 'ABEnv'});
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
plot(2, narrow_ABenvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(2, :), 'DisplayName', 'ABEnv'); % ABEnv point

% Connect points with a line for clarity
plot([1, 2], [narrow_EnvNormR_mean, narrow_ABenvNormR_mean], 'k--'); % Dashed line connecting points

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
plot(2, wide_ABenvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', colors(2, :), 'DisplayName', 'ABEnv'); % ABEnv point

% Connect points with a line for clarity
plot([1, 2], [wide_EnvNormR_mean, wide_ABenvNormR_mean], 'k--'); % Dashed line connecting points

% Customize the plot
title('Wide Condition: NormEnv vs ABEnv');
ylabel('Response Values');
xlim([0.5 2.5]); % Set X-axis limits
set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABEnv'}); % Set X-axis labels
grid on;


%%

%%%%%%% Ready Features
% % % % % Using different features to get Accuracies
% Combine other features (e.g., )
%%% Unused
% CmbOnsABenvConc = [ABenv(:,1:2) + Onset, ABenv(:,2:end)]; % ABEnvelope and Onset Concatinated (BinNum, all)
% % Com_Env_Onset_Concd = [stim_Env; stim_Onset]; % Env_Onset - Envelope and Onset Concatinated (2, all)
% % Com_Env_Onset_plus = stim_Env+ stim_Onset; % Env_Onset - Envelope and Onset Concatinated (2, all)
% % Com_OnsetPlusABenv = stim_ABenv + stim_Onset;
% % Ons_Env = [stim_Onset/30;stim_Env]; % Onset Envelope (Onset + Envelope) - Dividing to onset envelope to fractions to just emphasize onsets in Envelope
% % Normalized_Onset20x_ABenvelope = normalize(Onsetx23_plus_ABenvelope,2,'range'); % The Best so far
% % Norm_Env_Onset = Env_Onset ./ max(Env_Onset, [], 2); 
% OnsetConcABenvelope = [ABenv, Onset];
% feature_names = {'EnvNorm', 'ABenv', 'stim_Onset','CmbOnsABenvConc','OnsetConcABenvelope'};

%%%%%%% Onset Feature
% stim_Env = stim_Env';                                   % Transpose the envelope
% stim_Onset = OnsetGenerator(stim_Env); 
% 
% %%% Combine Top matrix of ABenvelope with Onset
% CmbOnsPlusTopABenv = stim_ABenv; CmbOnsPlusTopABenv(:,1) = stim_Onset;  % ABEnvelope and Onset Concatinated (BinNum, all)


%%%%%%% Check how it works - Thorge's ABenv model
% menv_norm = stim_Env;
% binEdges_dB = 8:9:64;  % Binning in 9 dB steps up to 64 dB (adjust as needed)
% [env_bin_norm_T, env_bin_T, binEdges_normalized_T, binEdges_dB_T] = ABenvelopeGenerator_V2_Thorge_modified(menv_norm, binEdges_dB)
% features_T = {menv_norm, env_bin_norm_T};









function Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvR_mean)

    % Define colors for the different conditions
    color_NormEnv = [0, 0.4470, 0.7410]; % Blue
    color_ABenv = [0.5 0 0.5]; % purple

    color_Lines = [0.2, 0.2, 0.2]; % Grey for connecting lines
    
    % Create a figure with better proportions
    figure('Position', [100, 100, 1000, 500]); 
   
    % Narrow Condition Plot (Left)
    subplot(1, 2, 1); % Create a subplot for Narrow condition
    hold on;

    % Boxplot for Narrow Condition
    h = boxplot([narrow_EnvNormR_mean, narrow_ABenvR_mean], ...
                'Labels', {'NormEnv', 'ABenv'}, 'Widths', 0.5);
    
    % Set properties for boxplot appearance
    set(h, 'LineWidth', 2); % Thicker lines for boxplot
    
    % Overlay points and lines for individual participants
    plot(1, narrow_EnvNormR_mean, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_NormEnv, ...
         'MarkerFaceColor', color_NormEnv, 'DisplayName', 'NormEnv'); % NormEnv points
    plot(2, narrow_ABenvR_mean, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_ABenv, ...
         'MarkerFaceColor', color_ABenv, 'DisplayName', 'ABenv'); % ABEnv points
     
    % Connect corresponding points with lines
    for i = 1:length(narrow_EnvNormR_mean)
        plot([1, 2], [narrow_EnvNormR_mean(i), narrow_ABenvR_mean(i)], 'Color', color_Lines, 'LineWidth', 1.2);
    end
    
    % Customize the plot
    title('Narrow: NormEnv vs ABenv', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Response Values', 'FontSize', 12);
    xlim([0.5 2.5]); % Set X-axis limits
    ylim([-0.03 0.19]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABenv'}, 'FontSize', 10);
    grid on;
    box on;

    % Wide Condition Plot (Right)
    subplot(1, 2, 2); % Create a subplot for Wide condition
    hold on;

    % Boxplot for Wide Condition
    h2 = boxplot([wide_EnvNormR_mean, wide_ABenvR_mean], ...
                 'Labels', {'NormEnv', 'ABenv'}, 'Widths', 0.5);
    
    % Set properties for boxplot appearance
    set(h2, 'LineWidth', 2); % Thicker lines for boxplot
    
    % Overlay points and lines for individual participants
    plot(1, wide_EnvNormR_mean, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_NormEnv, ...
         'MarkerFaceColor', color_NormEnv, 'DisplayName', 'NormEnv'); % NormEnv points
    plot(2, wide_ABenvR_mean, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_ABenv, ...
         'MarkerFaceColor', color_ABenv, 'DisplayName', 'ABenv'); % ABEnv points
     
    % Connect corresponding points with lines
    for i = 1:length(wide_EnvNormR_mean)
        plot([1, 2], [wide_EnvNormR_mean(i), wide_ABenvR_mean(i)], 'Color', color_Lines, 'LineWidth', 1.2);
    end
    
    % Customize the plot
    title('Wide: NormEnv vs ABenv', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Response Values', 'FontSize', 12);
    xlim([0.5 2.5]); % Set X-axis limits
    ylim([-0.03 0.19]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABenv'}, 'FontSize', 10);
    grid on;
    box on;

    % Tighten the layout and improve spacing
    set(gcf, 'Color', 'w'); % Set background color to white
    set(gca, 'FontSize', 12);
end






%%
% Plot Binned Envelopes
% figure;
% % Subplot 5: Combined Plot
% subplot(2, 1, 1);
% hold on;  % Hold for multiple plots
% % plot(time_audio, audio_dat, 'color','k', 'DisplayName', 'Original Audio');  % Original audio
% plot(time_env, EnvNorm, 'color', blue, 'DisplayName', 'Standard Envelope');  % Standard envelope
% plot(time_env, normSPLenvelope, 'color',orange, 'DisplayName', 'SPL Envelope');  % SPL envelope
% plot(time_env, norm_onsetEnvelope, 'color',yellow, 'DisplayName', 'Onset Envelope');  % Onset envelope
% 
% title('Combined Plot of Audio and Envelopes');
% xlabel('Time (s)');
% ylabel('Amplitude');
% xlim(window_time);
% legend('Location', 'best');  % Add legend
% grid on;  % Add grid for better visibility
% hold off;  % Release the hold
% 
% % %% Plot Binned Envelopes
% % figure;
% subplot(2, 1, 2);
% 
% for i = 1:numBins
%     plot_norm_binnedEnvelopes_dB = (norm_binnedEnvelopes_dB(:, i)*step_size) + binEdges_dB(i);
%     hold on
%     plot(time_env, plot_norm_binnedEnvelopes_dB);
%     title(['Binned Envelope (Bin ' num2str(i) ')']);
%     xlabel('Time (s)');
%     ylabel('Normalized Amplitude');
% end
% xlim(window_time);



%%
    % Define colors for the different conditions
    color_NormEnv = [0, 0.4470, 0.7410]; % Blue
    color_Combined = [0.1 0.5 0.1];
    color_ABenv = [0.5 0 0.5]; % purple

    color_Lines = [0.2, 0.2, 0.2]; % Grey for connecting lines
    
    % Create a figure with better proportions
    figure('Position', [100, 100, 1000, 500]); 
   
    % Narrow Condition Plot (Left)
    hold on;

    % Boxplot for Narrow Condition
    h = boxplot([meanABEnvR, meancombinedR], ...
                'Labels', {'ABenv', 'Combined'}, 'Widths', 0.5);
    
    % Set properties for boxplot appearance
    set(h, 'LineWidth', 2); % Thicker lines for boxplot
    
    % Overlay points and lines for individual participants
    plot(1, meanABEnvR, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_ABenv, ...
         'MarkerFaceColor', color_ABenv, 'DisplayName', 'Combined'); % NormEnv points
    plot(2, meancombinedR, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', color_Combined, ...
         'MarkerFaceColor', color_Combined, 'DisplayName', 'ABenv'); % ABEnv points
     
    % Connect corresponding points with lines
    for i = 1:length(meanABEnvR)
        plot([1, 2], [meanABEnvR(i), meancombinedR(i)], 'Color', color_Lines, 'LineWidth', 1.2);
    end
    
    % Customize the plot
    title('Average: (narrow+wide)/2: ABenv vs Combined', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Response Values', 'FontSize', 12);
    xlim([0.5 2.5]); % Set X-axis limits
    ylim([-0.03 0.19]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'ABenv', 'Combined(AB+Onset)'}, 'FontSize', 10);
    grid on;    box off;

    % Tighten the layout and improve spacing
    set(gcf, 'Color', 'w'); % Set background color to white
    set(gca, 'FontSize', 14);
