%% load 
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\Min_0_Max_6_Bin_4');
% name_struct_new; 

%%
% Initialize arrays to store Pearson r values
clear; clc; OT_setup

%%
% Define the folder with the files
folder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\';

% List all the .mat files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Initialize arrays to store results
significant_files = {};
non_significant_files = {};

% Get a list of all folders in the directory
folder_list = dir(folder_path);
folder_list = folder_list([folder_list.isdir] & ~startsWith({folder_list.name}, '.')); % Exclude '.' and '..'

%%
for i = 1:length(file_list)
    % Load the current file
    file_name = fullfile(folder_path, file_list(i).name);
    load(file_name);
    disp(file_name);
    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);

    % Test for normality
    normality_result = NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean,  wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);

    % Run Wilcoxon tests
    [p_val_narrow_ABenv, p_val_narrow_ABenvNorm, p_val_wide_ABenv, p_val_wide_ABenvNorm] = Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);

    % Label the files based on significance level (e.g., p < 0.05)
    if p_val_narrow_ABenv < 0.05 || p_val_narrow_ABenvNorm < 0.05 || p_val_wide_ABenv < 0.05 || p_val_wide_ABenvNorm < 0.05
        significant_files{end+1} = file_list(i).name;
    else
        non_significant_files{end+1} = file_list(i).name;
    end
end

% Display results
disp('Files with significant differences:');
disp(significant_files);

disp('Files without significant differences:');
disp(non_significant_files);
%%
%% 
% Define the folder with the files
folder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\';

% List all the .mat files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Initialize arrays to store results
significant_files = {};
non_significant_files = {};

% Get a list of all folders in the directory
folder_list = dir(folder_path);
folder_list = folder_list([folder_list.isdir] & ~startsWith({folder_list.name}, '.')); % Exclude '.' and '..'

% Create target folder paths for significant and non-significant files
target_folder_significant = fullfile(folder_path, '3.Significant Diff');
target_folder_non_significant = fullfile(folder_path, '1.No Significant Diff');

% Create the target folders if they don't exist
if ~exist(target_folder_significant, 'dir')
    mkdir(target_folder_significant);
end

if ~exist(target_folder_non_significant, 'dir')
    mkdir(target_folder_non_significant);
end


for i = 1:length(file_list)
    % Load the current file
    file_name = fullfile(folder_path, file_list(i).name);
    load(file_name);
    disp(file_name);
    
    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Test for normality
    normality_result = NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean,  wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);
    
    % Run Wilcoxon tests
    [p_val_narrow_ABenv, p_val_narrow_ABenvNorm, p_val_wide_ABenv, p_val_wide_ABenvNorm] = Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);
    
    % Label the files based on significance level (e.g., p < 0.05)
    if p_val_narrow_ABenv < 0.05 || p_val_narrow_ABenvNorm < 0.05 || p_val_wide_ABenv < 0.05 || p_val_wide_ABenvNorm < 0.05
        significant_files{end+1} = file_list(i).name;
        
        % Move the significant file to the target folder
        movefile(file_name, target_folder_significant);
    else
        non_significant_files{end+1} = file_list(i).name;
        
        % Move the non-significant file to the target folder
        movefile(file_name, target_folder_non_significant);
    end
end

% Display results
disp('Files with significant differences:');
disp(significant_files);

disp('Files without significant differences:');
disp(non_significant_files);
%%
%% Define the folder with significant files
folder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\3.Significant Diff\';

% List all the .mat files in the significant folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Define the paths for the existing subfolders
target_folder_AB_better_narrow = fullfile(folder_path, '4.ABbetter_narrow');
target_folder_AB_better_wide = fullfile(folder_path, '4.ABbetter_wide');
target_folder_standard_better = fullfile(folder_path, '5.ABworse');
target_folder_AB_better = fullfile(folder_path, '6.ABbetter');


% Create the subfolders if they don't exist
if ~exist(target_folder_AB_better_narrow, 'dir')
    mkdir(target_folder_AB_better_narrow);
end

if ~exist(target_folder_AB_better_wide, 'dir')
    mkdir(target_folder_AB_better_wide);
end

if ~exist(target_folder_standard_better, 'dir')
    mkdir(target_folder_standard_better);
end

% Classify files based on which method is better
for i = 1:length(file_list)
    % Load the current file
    file_name = fullfile(folder_path, file_list(i).name);
    load(file_name);
    
    % Perform the analysis (Assuming extraction_pearson is the same)
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, ...
        narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, ...
        wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = ...
        extraction_pearson(sbj, task, name_struct_new);
    
    % Check for NaN values
    if all(isnan(narrow_ABenvR_mean)) || all(isnan(narrow_EnvNormR_mean)) || ...
       all(isnan(wide_ABenvR_mean)) || all(isnan(wide_EnvNormR_mean))
        disp(['File ', file_list(i).name, ' contains NaN values in comparison variables.']);
        continue; % Skip this file
    end
    
    % Initialize flags for classification
    narrow_better = (mean(narrow_ABenvR_mean) > mean(narrow_EnvNormR_mean)) || ...
                     (mean(narrow_ABenvNormR_mean) > mean(narrow_EnvNormR_mean));
    wide_better = (mean(wide_ABenvR_mean) > mean(wide_EnvNormR_mean)) || ...
                   (mean(wide_ABenvNormR_mean) > mean(wide_EnvNormR_mean));

    % Classify based on which method is better
    if narrow_better && wide_better
        % Move to narrow and wide AB Better folders
        movefile(file_name, target_folder_AB_better);
        movefile(file_name, target_folder_AB_better);
    elseif narrow_better
        % Move to AB Better Narrow folder
        movefile(file_name, target_folder_AB_better_narrow);
    elseif wide_better
        % Move to AB Better Wide folder
        movefile(file_name, target_folder_AB_better_wide);
    elseif ~narrow_better && ~wide_better
        % Move to Standard Better folder
        movefile(file_name, target_folder_standard_better);
    else
        disp(['File ', file_list(i).name, ' has mixed results and cannot be classified.']);
    end
end

% Display completion message
disp('Classification of significant files into existing folders is complete.');




%%
function [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, ...
    narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new)


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
   	        if isfield(name_struct_new(s, k), 'ABenvNormStats')
                if strcmp(task{k}, 'narrow')
                    narrow_ABenvNormR_mean(s) = mean(name_struct_new(s, k).ABenvNormStats.r);
                elseif strcmp(task{k}, 'wide')
                    wide_ABenvNormR_mean(s) = mean(name_struct_new(s, k).ABenvNormStats.r);
                end
            end
            if isfield(name_struct_new(s, k), 'ABenvRedBinStats') && isstruct(name_struct_new(s, k).ABenvRedBinStats) && isfield(name_struct_new(s, k).ABenvRedBinStats, 'r')
                if strcmp(task{k}, 'narrow')
                    narrow_ABenvRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvRedBinStats.r);
                elseif strcmp(task{k}, 'wide')
                    wide_ABenvRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvRedBinStats.r);
                end
            end
            if isfield(name_struct_new(s, k), 'ABenvNormRedBinStats') && isstruct(name_struct_new(s, k).ABenvNormRedBinStats) && isfield(name_struct_new(s, k).ABenvNormRedBinStats, 'r')
                if strcmp(task{k}, 'narrow')
                    narrow_ABenvNormRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvNormRedBinStats.r);
                elseif strcmp(task{k}, 'wide')
                    wide_ABenvNormRedBinR_mean(s) = mean(name_struct_new(s, k).ABenvNormRedBinStats.r);
                end
    
            end
        end
    end

    % Inside the extraction_pearson function after computing narrow_ABenvRedBinR_mean
    if all(~isnan(narrow_ABenvRedBinR_mean))
        disp('Reduction is worked! narrow_ABenvRedBinR_mean and narrow_ABenvNormRedBinR_mean contains some values.');
    else
        disp('Reduction is not working: narrow_ABenvRedBinR_mean and narrow_ABenvNormRedBinR_mean contains only NaN values.');
    end

    % Inside the extraction_pearson function after computing narrow_ABenvRedBinR_mean
    if all(~isnan(wide_ABenvRedBinR_mean))
        disp('Reduction is worked! wide_ABenvRedBinR_mean and wide_ABenvNormRedBinR_mean contains some values.');
    else
        disp('Reduction is not working: wide_ABenvRedBinR_mean and wide_ABenvNormRedBinR_mean contains only NaN values.');
    end

end

%%
function normality_result = NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)
    % Add Shapiro-Wilk test
    addpath('C:\Users\icbmadmin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Functions\Shapiro-Wilk and Shapiro-Francia normality tests')
    
    % Shapiro-Wilk test
    features = {narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean};
    
    % Initialize the result to true (all normal)
    normality_result = true;
    
    for i = 1:length(features)
        % Shapiro-Wilk test for each feature
        [~, p_sw, ~] = swtest(features{i}, 0.05);
        
        if p_sw > 0.05
            disp(['Shapiro-Wilk Test p-value = ' num2str(p_sw) ' - Normally distributed']);
        else
            disp(['Shapiro-Wilk Test p-value = ' num2str(p_sw) ' - Not normally distributed']);
            % If any feature is not normally distributed, set the result to false
            normality_result = false;
        end
    end
end

%%
function Plot_PearsonR(sbj, narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean)
    % Plot Pearson r values for narrow condition
    figure;
    subplot(2,1,1)
    hold on;
    plot(1:length(sbj), narrow_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Narrow)');
    plot(1:length(sbj), narrow_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Narrow)');
    % plot(1:length(sbj), narrow_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Narrow)');
    plot(1:length(sbj), narrow_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins)');
    % plot(1:length(sbj), narrow_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins)');
    
    
    xlabel('Subject');
    ylabel('Pearson r (mean)');
    title('Model NormEnv vs ABEnv - Narrow Condition');
    legend(location='best');
    ylim([-0.05 0.20])
    grid on;
    % Save the plot
    % fig_filename_narrow = sprintf('PearsonsR_Narrow');
    % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_narrow)
    
    
    % Plot Pearson r values for wide condition
    % figure;
    subplot(2,1,2)
    hold on;
    plot(1:length(sbj), wide_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Wide)');
    plot(1:length(sbj), wide_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Wide)');
    % plot(1:length(sbj), wide_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Wide)');
    plot(1:length(sbj), wide_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins, Wide)');
    % plot(1:length(sbj), wide_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins, Wide)');
    xlabel('Subject');
    ylabel('Pearson r (mean)');
    title('Model NormEnv vs ABEnv - Wide Condition');
    legend(location='best');
    grid on;
    ylim([-0.05 0.20])
    % Save the plot
    % fig_filename_wide = sprintf('PearsonsR_Wide');
    % save_fig(gcf, 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figures\PearsonsRall\',fig_filename_wide)

end

%% Wilcoxon_run
function [p_wilcoxon_Narrow, p_wilcoxon_Narrow_norm, p_wilcoxon_Wide, p_wilcoxon_Wide_norm] = Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)
    % Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnv
    [p_wilcoxon_Narrow, ~] = signrank(narrow_EnvNormR_mean, narrow_ABenvR_mean, 'alpha', 0.05);
    if p_wilcoxon_Narrow < 0.05
        disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv)     p-value = ' num2str(p_wilcoxon_Narrow) ' - Significant difference']);
    else
        disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnv)     p-value = ' num2str(p_wilcoxon_Narrow) ' - No significant difference']);
    end

    % Wilcoxon signed-rank test for Narrow condition: NormEnv vs ABEnvNorm
    [p_wilcoxon_Narrow_norm, ~] = signrank(narrow_EnvNormR_mean, narrow_ABenvNormR_mean, 'alpha', 0.05);
    if p_wilcoxon_Narrow_norm < 0.05
        disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_wilcoxon_Narrow_norm) ' - Significant difference']);
    else
        disp(['Condition: Narrow ; Wilcoxon signed-rank Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_wilcoxon_Narrow_norm) ' - No significant difference']);
    end

    % Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnv
    [p_wilcoxon_Wide, ~] = signrank(wide_EnvNormR_mean, wide_ABenvR_mean, 'alpha', 0.05);
    if p_wilcoxon_Wide < 0.05
        disp(['Condition: Wide   ; Wilcoxon signed-rank Test (NormEnv vs ABEnv)     p-value = ' num2str(p_wilcoxon_Wide) ' - Significant difference']);
    else
        disp(['Condition: Wide   ; Wilcoxon signed-rank Test (NormEnv vs ABEnv)     p-value = ' num2str(p_wilcoxon_Wide) ' - No significant difference']);
    end

    % Wilcoxon signed-rank test for Wide condition: NormEnv vs ABEnvNorm
    [p_wilcoxon_Wide_norm, ~] = signrank(wide_EnvNormR_mean, wide_ABenvNormR_mean, 'alpha', 0.05);
    if p_wilcoxon_Wide_norm < 0.05
        disp(['Condition: Wide   ; Wilcoxon signed-rank Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_wilcoxon_Wide_norm) ' - Significant difference']);
    else
        disp(['Condition: Wide   ; Wilcoxon signed-rank Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_wilcoxon_Wide_norm) ' - No significant difference']);
    end

end

%% Box_and_Ranked_Plot
function Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)

    % Create a figure for boxplots
    figure;
    
    % Narrow condition
    subplot(2, 2, 1); % Create a subplot for Narrow condition
    boxplot([narrow_ABenvR_mean, narrow_EnvNormR_mean,  narrow_ABenvNormR_mean], 'Labels', {'ABenv', 'NormEnv', 'ABenvNorm'});
    title('Narrow: NormEnv vs ABEnv vs ABenvNorm');
    ylabel('Response Values');
    ylim([-0.05 0.2]);
    grid on;
    

    % Wide condition
    subplot(2, 2, 2); % Create a subplot for Wide condition
    boxplot([wide_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvNormR_mean], 'Labels', {'ABenv', 'NormEnv', 'ABenvNorm'});
    title('Wide: NormEnv vs ABEnv vs ABenvNorm');
    ylabel('Response Values');
    ylim([-0.05 0.2]);
    grid on;
    
    
    
    % figure;
    % Narrow condition
    subplot(2, 2, 3); % Create a subplot for Narrow condition
    hold on;
    
    % Plotting the first participant's data for Narrow Condition
    plot(1, narrow_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'ABenv'); % ABEnv point
    plot(2, narrow_EnvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'NormEnv'); % NormEnv point
    plot(3, narrow_ABenvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'ABenvNorm'); % ABEnvNorm point
    
    % Connect points with a line for clarity
    plot([1, 2, 3], [narrow_ABenvR_mean, narrow_EnvNormR_mean, narrow_ABenvNormR_mean], 'k--'); % Dashed line connecting points
    
    % Customize the plot
    title('Narrow: NormEnv vs ABenv vs ABenvNorm');
    ylabel('Response Values');
    xlim([0.5 3.5]); % Set X-axis limits
    ylim([-0.05 0.2]);

    set(gca, 'XTick', [1 2 3], 'XTickLabel', { 'ABenv', 'NormEnv','ABenvNorm'}); % Set X-axis labels
    grid on;
    
    
    % Wide condition
    subplot(2, 2, 4); % Create a subplot for Wide condition
    hold on;
    
    % Plotting the first participant's data for Wide Condition
    plot(1, wide_ABenvR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'ABenv'); % ABenv point
    plot(2, wide_EnvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'NormEnv'); % NormEnv point
    plot(3, wide_ABenvNormR_mean, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'ABenvNorm'); % ABenvNorm point
    
    % Connect points with a line for clarity
    plot([1, 2, 3], [wide_ABenvR_mean, wide_EnvNormR_mean, wide_ABenvNormR_mean], 'k--'); % Dashed line connecting points
    
    % Customize the plot
    title('Wide: NormEnv vs ABEnv vs ABenvNorm');
    ylabel('Response Values');
    xlim([0.5 3.5]); % Set X-axis limits
    ylim([-0.05 0.2]);
    set(gca, 'XTick', [1 2 3], 'XTickLabel', {'ABenv', 'NormEnv', 'ABenvNorm'}); % Set X-axis labels
    grid on;
end

%%

% Define parameter ranges
min_bin_values = [0, 10, 20, 30, 40, 50, 60, 70];
max_bin_values = [6, 10, 20, 30, 40, 50, 60, 70, 80];
num_bins_values = [4, 5, 6, 8, 12, 16];

% Generate all combinations of parameters where max_bin > min_bin
all_combinations = [];
for min_bin = min_bin_values
    for max_bin = max_bin_values
        if max_bin > min_bin % Only consider combinations where max_bin > min_bin
            for num_bins = num_bins_values
                all_combinations = [all_combinations; min_bin, max_bin, num_bins];
            end
        end
    end
end

% Define folder paths
ABworsefolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\3.Significant Diff\5.ABworse\';
ABBetterfolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\3.Significant Diff\4.ABbetter_narrow\'; 
Nosignificantfolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\1.No Significant Diff';

% List all .mat files in the folders
ABworse_file_list = dir(fullfile(ABworsefolder_path, '*.mat'));
ABBetter_file_list = dir(fullfile(ABBetterfolder_path, '*.mat'));
Nosignificant_file_list = dir(fullfile(Nosignificantfolder_path, '*.mat'));

% Prepare to extract parameter values from filenames for AB worse
ab_worse_configs = zeros(length(ABworse_file_list), 3); % For min_bin, max_bin, num_bins
for i = 1:length(ABworse_file_list)
    parts = strsplit(ABworse_file_list(i).name, {'_', '.'}); % Access name using parentheses
    min_bin = str2double(parts{2});
    max_bin = str2double(parts{4});
    num_bins = str2double(parts{6});
    ab_worse_configs(i, :) = [min_bin, max_bin, num_bins];
end

% Prepare to extract parameter values from filenames for AB better
ab_better_configs = zeros(length(ABBetter_file_list), 3); % For min_bin, max_bin, num_bins
for i = 1:length(ABBetter_file_list)
    parts = strsplit(ABBetter_file_list(i).name, {'_', '.'}); % Access name using parentheses
    min_bin = str2double(parts{2});
    max_bin = str2double(parts{4});
    num_bins = str2double(parts{6});
    ab_better_configs(i, :) = [min_bin, max_bin, num_bins];
end

% Prepare to extract parameter values from filenames for no significance
ab_nosignificance_configs = zeros(length(Nosignificant_file_list), 3); % For min_bin, max_bin, num_bins
for i = 1:length(Nosignificant_file_list)
    parts = strsplit(Nosignificant_file_list(i).name, {'_', '.'}); % Access name using parentheses
    min_bin = str2double(parts{2});
    max_bin = str2double(parts{4});
    num_bins = str2double(parts{6});
    ab_nosignificance_configs(i, :) = [min_bin, max_bin, num_bins];
end

% Create a 3D scatter plot of all combinations in black
figure;
scatter3(all_combinations(:, 1), all_combinations(:, 2), all_combinations(:, 3), 50, 'k', 'filled', 'Marker', 'o');
hold on;

% Overlay AB worse configurations in red
scatter3(ab_worse_configs(:, 1), ab_worse_configs(:, 2), ab_worse_configs(:, 3), 100, 'r', 'filled', 'Marker', 'o');

% Overlay AB better configurations in green
scatter3(ab_better_configs(:, 1), ab_better_configs(:, 2), ab_better_configs(:, 3), 100, 'g', 'filled', 'Marker', 'o');

% Overlay no significance configurations in yellow
scatter3(ab_nosignificance_configs(:, 1), ab_nosignificance_configs(:, 2), ab_nosignificance_configs(:, 3), 100, 'y', 'filled', 'Marker', 'o');

% Customize plot
xlabel('Min Bin Values');
ylabel('Max Bin Values');
zlabel('Number of Bins');
title('3D Scatter Plot of Parameter Combinations');
grid on; % Enable grid for better reference

legend('All Combinations', 'AB Worse', 'AB Better', 'No Significance', 'Location', 'best');

% Set limits for each axis
xlim([min(min_bin_values), max(min_bin_values)]);
ylim([min(max_bin_values), max(max_bin_values)]);
zlim([min(num_bins_values), max(num_bins_values)]);

% Set view angle for better visibility
view(30, 30);
hold off;

%%
% clear; 
clc; OT_setup
%%
% Define the directory containing the results
folder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoopSignificant\3.Significant Diff\'; 

% List all the .mat files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Initialize a cell array to store the results
results = cell(length(file_list), 11); % Pre-allocate for efficiency
% Columns: [File Name, Narrow EnvNormR Mean, Wide EnvNormR Mean, Narrow ABenvR Mean, Wide ABenvR Mean, Narrow ABenvNormR Mean, Wide ABenvNormR Mean, Narrow ABenvRedBinR Mean, Wide ABenvRedBinR Mean, Narrow ABenvNormRedBinR Mean, Wide ABenvNormRedBinR Mean]

% Loop through each file and perform the analysis
for i = 1:length(file_list)
    % Load the current file
    file_name = fullfile(folder_path, file_list(i).name);
    load(file_name);
    disp(['Processing file: ', file_list(i).name]); % Display the file name for tracking

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, ...
        narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, ...
        narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);

    % Store mean values in the results array
    results{i, 1} = file_list(i).name; % File name
    results{i, 2} = mean(narrow_EnvNormR_mean); % Narrow EnvNormR Mean
    results{i, 3} = mean(wide_EnvNormR_mean); % Wide EnvNormR Mean
    results{i, 4} = mean(narrow_ABenvR_mean); % Narrow ABenvR Mean
    results{i, 5} = mean(wide_ABenvR_mean); % Wide ABenvR Mean
    results{i, 6} = mean(narrow_ABenvNormR_mean); % Narrow ABenvNormR Mean
    results{i, 7} = mean(wide_ABenvNormR_mean); % Wide ABenvNormR Mean

    % Check if values are not NaN and store means for additional parameters
    if ~any(isnan(narrow_ABenvRedBinR_mean)) && ~any(isnan(wide_ABenvRedBinR_mean))
        results{i, 8} = mean(narrow_ABenvRedBinR_mean); % narrow_ABenvRedBinR_mean
        results{i, 9} = mean(wide_ABenvRedBinR_mean); % wide_ABenvRedBinR_mean
    else
        results{i, 8} = NaN; % Set to NaN if input is NaN
        results{i, 9} = NaN; % Set to NaN if input is NaN
    end
    
    if ~any(isnan(narrow_ABenvNormRedBinR_mean)) && ~any(isnan(wide_ABenvNormRedBinR_mean))
        results{i, 10} = mean(narrow_ABenvNormRedBinR_mean); % narrow_ABenvNormRedBinR_mean
        results{i, 11} = mean(wide_ABenvNormRedBinR_mean); % wide_ABenvNormRedBinR_mean
    else
        results{i, 10} = NaN; % Set to NaN if input is NaN
        results{i, 11} = NaN; % Set to NaN if input is NaN
    end
end

% Convert results to a table for better visualization
results_table = cell2table(results, 'VariableNames', ...
    {'FileName', 'Narrow_EnvNormR_Mean', 'Wide_EnvNormR_Mean', ...
     'Narrow_ABenvR_Mean', 'Wide_ABenvR_Mean', ...
     'Narrow_ABenvNormR_Mean', 'Wide_ABenvNormR_Mean', ...
     'Narrow_ABenvRedBinR_Mean', 'Wide_ABenvRedBinR_Mean', ...
     'Narrow_ABenvNormRedBinR_Mean', 'Wide_ABenvNormRedBinR_Mean'});

% Display the results table
disp(results_table);


% Optionally, you can save the results to a file
% writetable(results_table, fullfile(folder_path, 'Analysis_Results.csv'));

%%
find(results_table.Narrow_ABenvR_Mean==max(results_table.Narrow_ABenvR_Mean))
find(results_table.Wide_ABenvR_Mean==max(results_table.Wide_ABenvR_Mean))
find(results_table.Narrow_ABenvRedBinR_Mean==max(results_table.Narrow_ABenvRedBinR_Mean))
find(results_table.Wide_ABenvRedBinR_Mean==max(results_table.Wide_ABenvRedBinR_Mean))

%%

%%
% clear; clc; OT_setup
%%
[narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, ...
    narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);

NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean,  wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)

Plot_PearsonR(sbj, narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean)

% % 3 Features - Without Bin Reduction
[p_wilcoxon_Narrow, p_wilcoxon_Narrow_norm, p_wilcoxon_Wide, p_wilcoxon_Wide_norm] = Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);

Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)

%% 5 Features - With Bin Reduction
% Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvRedBinR_mean)
% Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvRedBinR_mean)
%% 5 Features - With norm Bin Reduction
Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormRedBinR_mean)
Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormRedBinR_mean)
