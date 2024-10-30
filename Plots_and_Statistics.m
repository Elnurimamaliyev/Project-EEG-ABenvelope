%% Cleaning
clear; clc;
%% Add Main paths
OT_setup
%%
% Initialize arrays to store Pearson r values
% Narrow
narrow_EnvNormR_mean = NaN(length(sbj), 1);
narrow_ABenvR_mean = NaN(length(sbj), 1);
narrow_ABenvRedBinR_mean = NaN(length(sbj), 1);

% Wide
wide_EnvNormR_mean = NaN(length(sbj), 1);
wide_ABenvR_mean = NaN(length(sbj), 1);
wide_ABenvRedBinR_mean = NaN(length(sbj), 1);

DbCounts_all = cell(length(sbj), length(task)); 
binEdges_dB_all = cell(length(sbj), length(task));

% Extract Pearson r values for narrow and wide conditions
for s = 1:length(sbj)
    for k = 1:length(task)
        % Extract DbCounts and binEdges_dB for the current subject and task
        % DbCounts_all{s, k} = name_struct_new(s, k).DbCounts;
        % binEdges_dB_all{s, k} = name_struct_new(s, k).binEdges_dB;

        if isfield(name_struct_new(s, k), 'EnvNormStats')
            if strcmp(task{k}, 'narrow') % Assuming task names are used to differentiate conditions
                narrow_EnvNormR_mean(s) = mean(name_struct_new(s, k).EnvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_EnvNormR_mean(s) = mean(name_struct_new(s, k).EnvNormStats.r);
            end
        end
        if isfield(name_struct_new(s, k), 'ABenvStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvR_mean(s) = mean(name_struct_new(s, k).ABenvStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvR_mean(s) = mean(name_struct_new(s, k).ABenvStats.r);
            end
        end
        if isfield(name_struct_new(s, k), 'ABenvRedBinStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvRedBinStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvRedBinStats.r);
            end
        end

    end
end

% Plot Pearson r values
% Sort increasing order
[sorted_narrow_EnvNormR_mean, sortIdx_narrow] = sort(narrow_EnvNormR_mean, 'ascend');
% Apply the same sorting
sorted_narrow_ABenvR_mean = narrow_ABenvR_mean(sortIdx_narrow);

[sorted_wide_EnvNormR_mean, sortIdx_wide] = sort(wide_EnvNormR_mean, 'ascend');
% Apply the same sorting
sorted_wide_ABenvR_mean = wide_ABenvR_mean(sortIdx_wide);

% Chance level
chance_level = 0.005; 

figure;
% Narrow Condition
subplot(1,2,1); hold on;
plot(1:length(sbj), sorted_narrow_EnvNormR_mean, 'o-', 'DisplayName', 'Env', 'LineWidth',2, 'MarkerSize', 5); % Thicker line and smaller markers
plot(1:length(sbj), sorted_narrow_ABenvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'ABEnv', 'LineWidth', 2, 'MarkerSize', 5); % Thicker line and smaller markers
xlabel('Subjects (sorted by ClassicEnv values)', 'FontWeight', 'bold'); 
ylabel("Pearsons' r (mean)");
% ylabel("Pearsons' r (mean)", 'FontWeight', 'bold');
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);

xlim([0.5 20.5]); 
title('Narrow Condition', 'FontWeight', 'bold'); 
grid on; hold on;
yline(chance_level, '--k', 'DisplayName', 'Chance Level (+0.005)', 'LineWidth', 1.5); % Optional: Increase line width for visibility
yline(-chance_level, '--k', 'DisplayName', 'Chance Level (-0.005)', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 
% Set tick label font size for the first subplot
set(gca, 'FontSize', 13); % Adjust the font size for tick labels


% Wide Condition
subplot(1,2,2); hold on;
plot(1:length(sbj), sorted_wide_EnvNormR_mean, 'o-', 'DisplayName', 'Env', 'LineWidth', 2, 'MarkerSize', 5); % Thicker line and smaller markers
plot(1:length(sbj), sorted_wide_ABenvR_mean, 'o-', 'Color', [0.5 0 0.5], 'DisplayName', 'ABEnv', 'LineWidth', 2, 'MarkerSize', 5); 
xlabel('Subjects (sorted by ClassicEnv values)', 'FontWeight', 'bold', 'FontSize', 12); 
ylabel("Pearsons' r (mean)");
ylim([-0.02 0.20]); yticks(-0.02:0.04:0.20);

xlim([0 21]); 
title('Wide Condition', 'FontWeight', 'bold'); 
grid on; hold on;
yline(chance_level, '--k', 'DisplayName', 'Chance Level (+0.005)', 'LineWidth', 1.5); % Optional: Increase line width for visibility
yline(-chance_level, '--k', 'DisplayName', 'Chance Level (-0.005)', 'LineWidth', 1.5);
legend('show', 'Location', 'northwest'); 

set(gca, 'FontSize', 13); % Adjust the font size for tick labels

% Save the plot
% fig_filename = sprintf('PearsonsR_max_itself');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename)
%% Box plot
Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvR_mean)
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
        DbCounts_all_narrow(s,:) = StatsParticipantTask_0(s, 1).DbCounts;
        DbCounts_all_wide(s,:) = StatsParticipantTask_0(s, 2).DbCounts;
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
variables = {narrow_EnvNormR_mean, narrow_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvR_mean};
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
% Paired T test for Narrow condition: NormEnv vs ABEnv
[h_ttest_Narrow, p_ttest_Narrow] = ttest(narrow_ABenvRedBinR_mean, narrow_ABenvR_mean);
if p_ttest_Narrow < 0.05
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Narrow) ' - Significant difference']);
else
    disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv) p-value = ' num2str(p_ttest_Narrow) ' - No significant difference']);
end

% Paired T test for Wide condition: NormEnv vs ABEnv
[h_ttest_Wide,p_ttest_Wide] = ttest(wide_ABenvRedBinR_mean, wide_ABenvR_mean);
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




%% Topoplot
%%% plot that uses pearson r of normal envelope model and ab envelope over each subject
% clear
OT_setup
% addpath("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data")
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_new.mat');
% %%
[~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task);

chanlocsData = EEG.chanlocs;
figure; 
[matrix_StdEnv_Narrow, matrix_ABenv_Narrow, matrix_StdEnv_Wide, matrix_ABenv_Wide] = PlotTopoplot(name_struct_new, chanlocsData, sbj);
%%
set(gca, 'FontSize', 15); % Adjust the font size for tick labels
%%
hFigure = gcf
hFigure.Color = [1,1,1]
hAxes = findobj(gcf,"Type","axes")
hAxes(1).Color = [1,1,1]
% Topoplot_filename = sprintf('Topoplota');
% save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\Topoplot\',Topoplot_filename)

%%
function Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvR_mean)

    % Define colors for the different conditions
    color_ABenv = [0.8500, 0.3250, 0.0980]; % Red
    color_NormEnv = [0, 0.4470, 0.7410]; % Blue
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