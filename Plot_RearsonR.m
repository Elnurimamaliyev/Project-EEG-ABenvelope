
% load 
name_struct = [];

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
        if isfield(name_struct(s, k), 'EnvNormStats')
            if strcmp(task{k}, 'narrow') % Assuming task names are used to differentiate conditions
                narrow_EnvNormR_mean(s) = mean(name_struct(s, k).EnvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_EnvNormR_mean(s) = mean(name_struct(s, k).EnvNormStats.r);
            end
        end
        if isfield(name_struct(s, k), 'ABenvStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvR_mean(s) = mean(name_struct(s, k).ABenvStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvR_mean(s) = mean(name_struct(s, k).ABenvStats.r);
            end
        end
   	    if isfield(name_struct(s, k), 'ABenvNormStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvNormR_mean(s) = mean(name_struct(s, k).ABenvNormStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvNormR_mean(s) = mean(name_struct(s, k).ABenvNormStats.r);
            end
        end
        if isfield(name_struct(s, k), 'ABenvRedBinStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvRedBinR_mean(s) = mean(name_struct(s, k).ABenvRedBinStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvRedBinR_mean(s) = mean(name_struct(s, k).ABenvRedBinStats.r);
            end
        end
        if isfield(name_struct(s, k), 'ABenvNormRedBinStats')
            if strcmp(task{k}, 'narrow')
                narrow_ABenvNormRedBinR_mean(s) = mean(name_struct(s, k).ABenvNormRedBinStats.r);
            elseif strcmp(task{k}, 'wide')
                wide_ABenvNormRedBinR_mean(s) = mean(name_struct(s, k).ABenvNormRedBinStats.r);
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
% Shapiro-Wilk test for Narrow condition: ABEnv
[h_sw_ABEnv, p_sw_ABEnv, stats_sw_ABEnv] = swtest(narrow_ABenvNormRedBinR_mean, 0.05);
if p_sw_ABEnv > 0.05
    disp(['Participant ' num2str(s) ', Condition: Narrow ;  Shapiro-Wilk Test (ABEnv) p-value = ' num2str(p_sw_ABEnv) ' - Normally distributed']);
else
    disp(['Participant ' num2str(s) ', Condition: Narrow ;  Shapiro-Wilk Test (ABEnv) p-value = ' num2str(p_sw_ABEnv) ' - Not normally distributed']);
end
%%
narrow_ABenvNormR_mean;
narrow_ABenvRedBinR_mean;
narrow_ABenvNormRedBinR_mean;
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
     'DisplayName', 'NormEnv'); % NormEnv point
plot(2, narrow_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'ABEnv'); % ABEnv point

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
     'DisplayName', 'NormEnv'); % NormEnv point
plot(2, wide_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'ABEnv'); % ABEnv point

% Connect points with a line for clarity
plot([1, 2], [wide_EnvNormR_mean, wide_ABenvR_mean], 'k--'); % Dashed line connecting points

% Customize the plot
title('Wide Condition: NormEnv vs ABEnv');
ylabel('Response Values');
xlim([0.5 2.5]); % Set X-axis limits
set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABEnv'}); % Set X-axis labels
grid on;