%% load 
% load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\Min_0_Max_6_Bin_4');
% name_struct_new; 

%%
% Initialize arrays to store Pearson r values
clear; clc; OT_setup
%%
% load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\FineTuning_tuned\TSignificant\Min_0_Max_10_Bin_8.mat")
% %%
% load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\FineTuning_tuned\TSignificant\Min_1_Max_7_Bin_7.mat")
%%
% Initialize arrays to store Pearson r values
[narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, ...
    narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
%%
% % Normality tests
% [h_narrow_norm, p_narrow_norm] = swtest(narrow_EnvNormR_mean);
% [h_narrow_ab, p_narrow_ab] = swtest(narrow_ABenvR_mean);
% [h_wide_norm, p_wide_norm] = swtest(wide_EnvNormR_mean);
% [h_wide_ab, p_wide_ab] = swtest(wide_ABenvR_mean);
% 
% % Display normality results
% fprintf('Normality Test (Narrow NormEnv): h = %d, p = %.4f\n', h_narrow_norm, p_narrow_norm);
% fprintf('Normality Test (Narrow ABenv): h = %d, p = %.4f\n', h_narrow_ab, p_narrow_ab);
% fprintf('Normality Test (Wide NormEnv): h = %d, p = %.4f\n', h_wide_norm, p_wide_norm);
% fprintf('Normality Test (Wide ABenv): h = %d, p = %.4f\n', h_wide_ab, p_wide_ab);

% Paired-sample T-Test comparing narrow NormEnv vs. narrow ABenv
[h_narrow, p_narrow] = ttest(narrow_EnvNormR_mean, narrow_ABenvR_mean);
fprintf('Paired T-Test for Narrow NormEnv vs. ABenv: h = %d, p = %.4f\n', h_narrow, p_narrow);

% Significance statement for narrow conditions
if p_narrow < 0.05
    fprintf('The difference between Narrow NormEnv and ABenv is statistically significant (p < 0.05).\n');
else
    fprintf('The difference between Narrow NormEnv and ABenv is not statistically significant (p >= 0.05).\n');
end

% Paired-sample T-Test comparing wide NormEnv vs. wide ABenv
[h_wide, p_wide] = ttest(wide_EnvNormR_mean, wide_ABenvR_mean);
fprintf('Paired T-Test for Wide NormEnv vs. ABenv: h = %d, p = %.4f\n', h_wide, p_wide);

% Significance statement for wide conditions
if p_wide < 0.05
    fprintf('The difference between Wide NormEnv and ABenv is statistically significant (p < 0.05).\n');
else
    fprintf('The difference between Wide NormEnv and ABenv is not statistically significant (p >= 0.05).\n');
end
%%
% Define the folder with the files
folder_path = 'P:\AllUsers\save'; 

% List all the .mat files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Initialize arrays to store results
significant_files = {};
non_significant_files = {};

% Create target folder paths for significant and non-significant files
target_folder_significant = fullfile(folder_path, 'TSignificant');
target_folder_non_significant = fullfile(folder_path, 'TnonSignificant');

% Create the target folders if they don't exist
if ~exist(target_folder_significant, 'dir')
    mkdir(target_folder_significant);
end

if ~exist(target_folder_non_significant, 'dir')
    mkdir(target_folder_non_significant);
end

% Loop through each file
for i = 1:length(file_list)
    % Load the current file
    file_name = fullfile(folder_path, file_list(i).name);
    try
        load(file_name);
        disp(['Processing: ', file_name]);

        % Perform the analysis
        [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
            narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
        
        % Test for normality
        normality_result = NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);
        
        % Run T-tests
        [h_narrow_ABenv, p_narrow_ABenv] = ttest(narrow_EnvNormR_mean, narrow_ABenvR_mean);
        [h_narrow_ABenvNorm, p_narrow_ABenvNorm] = ttest(narrow_EnvNormR_mean, narrow_ABenvNormR_mean);
        [h_wide_ABenv, p_wide_ABenv] = ttest(wide_EnvNormR_mean, wide_ABenvR_mean);
        [h_wide_ABenvNorm, p_wide_ABenvNorm] = ttest(wide_EnvNormR_mean, wide_ABenvNormR_mean);
        
        % Check significance for any p-values
        if any([p_narrow_ABenv, p_narrow_ABenvNorm, p_wide_ABenv, p_wide_ABenvNorm] < 0.05)
            disp([file_name, 'is significant!']);
            significant_files{end+1} = file_list(i).name;
            % Move the significant file to the target folder
            movefile(file_name, target_folder_significant);
        else
            non_significant_files{end+1} = file_list(i).name;
            % Move the non-significant file to the target folder
            movefile(file_name, target_folder_non_significant);
        end

    catch ME
        disp(['Error processing file ', file_list(i).name, ': ', ME.message]);
    end
end

% Display results
disp('Files with significant differences:');
disp(significant_files);

disp('Files without significant differences:');
disp(non_significant_files);

%%
% Create target folder paths for better performance conditions
target_folder_narrow_better = fullfile(folder_path, 'ABenvBetterNarrow');
target_folder_wide_better = fullfile(folder_path, 'ABenvBetterWide');
target_folder_both_better = fullfile(folder_path, 'ABenvBetterBoth');

% Create the target folders if they don't exist
if ~exist(target_folder_narrow_better, 'dir')
    mkdir(target_folder_narrow_better);
end

if ~exist(target_folder_wide_better, 'dir')
    mkdir(target_folder_wide_better);
end

if ~exist(target_folder_both_better, 'dir')
    mkdir(target_folder_both_better);
end

% Initialize arrays to store results
ABenv_better_narrow_files = {};
ABenv_better_wide_files = {};
ABenv_better_both_files = {};

% Loop through significant files
for i = 1:length(significant_files)
    % Load the significant file
    file_name = fullfile(target_folder_significant, significant_files{i});
    load(file_name);
    disp(['Analyzing: ', file_name]);

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Calculate the mean differences for narrow and wide conditions
    mean_diff_narrow = mean(narrow_ABenvR_mean - narrow_EnvNormR_mean);
    mean_diff_wide = mean(wide_ABenvR_mean - wide_EnvNormR_mean);
    
    % Check if ABenv is better than ClassicEnv in narrow, wide, or both conditions
    if mean_diff_narrow > 0 && mean_diff_wide > 0
        ABenv_better_both_files{end+1} = significant_files{i}; % ABenv better in both narrow and wide
        % Move the file to "ABenvBetterBoth" folder
        % movefile(file_name, target_folder_both_better);
        
    elseif mean_diff_narrow > 0
        ABenv_better_narrow_files{end+1} = significant_files{i}; % ABenv better in narrow
        % Move the file to "ABenvBetterNarrow" folder
        % movefile(file_name, target_folder_narrow_better);
        
    elseif mean_diff_wide > 0
        ABenv_better_wide_files{end+1} = significant_files{i}; % ABenv better in wide
        % Move the file to "ABenvBetterWide" folder
        % movefile(file_name, target_folder_wide_better);
    end
end

% Display results
disp('Files where ABenv is better in both narrow and wide:');
disp(ABenv_better_both_files);

disp('Files where ABenv is better in narrow:');
disp(ABenv_better_narrow_files);

disp('Files where ABenv is better in wide:');
disp(ABenv_better_wide_files);



%%
OT_setup
%%
folder_path_plot = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoopAllTtest\TSignificant\ABenvBetterNarrow\';
% Create figure for plotting narrow and wide comparisons
file_list_plot = dir(fullfile(folder_path_plot, '*.mat'));

for i = 1:length(file_list_plot)
    % Load the file
    file_name_plot = fullfile(folder_path_plot, file_list_plot(i).name);    
    % load(file_name_plot);

    % disp(['Plotting data for: ', file_name_plot]);
    load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoopAllTtest\TSignificant\ABenvBetterNarrow\Min_0_Max_10_Bin_8.mat")

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Sort narrow_EnvNormR_mean in increasing order
    [sorted_narrow_EnvNormR_mean, sortIdx_narrow] = sort(narrow_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to narrow_ABenvR_mean
    sorted_narrow_ABenvR_mean = narrow_ABenvR_mean(sortIdx_narrow);
    
    % Sort wide_EnvNormR_mean in increasing order
    [sorted_wide_EnvNormR_mean, sortIdx_wide] = sort(wide_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to wide_ABenvR_mean
    sorted_wide_ABenvR_mean = wide_ABenvR_mean(sortIdx_wide);
    
    % Plot narrow condition
    figure;
    subplot(2, 1, 1);
    plot(1:length(sorted_narrow_EnvNormR_mean), sorted_narrow_EnvNormR_mean, '-o', 'DisplayName', 'ClassicEnv (Narrow)');
    hold on;
    plot(1:length(sorted_narrow_ABenvR_mean), sorted_narrow_ABenvR_mean, '-x', 'DisplayName', 'ABenv (Narrow)');
    xlabel('Subjects (sorted by ClassicEnv values)');
    ylabel('Mean Pearson Correlation');
    title('Narrow Condition: ClassicEnv vs. ABenv (Sorted by ClassicEnv)');
    legend;
    grid on;
    
    % Plot wide condition
    subplot(2, 1, 2);
    plot(1:length(sorted_wide_EnvNormR_mean), sorted_wide_EnvNormR_mean, '-o', 'DisplayName', 'ClassicEnv (Wide)');
    hold on;
    plot(1:length(sorted_wide_ABenvR_mean), sorted_wide_ABenvR_mean, '-x', 'DisplayName', 'ABenv (Wide)');
    xlabel('Subjects (sorted by ClassicEnv values)');
    ylabel('Mean Pearson Correlation');
    title('Wide Condition: ClassicEnv vs. ABenv (Sorted by ClassicEnv)');
    legend;
    grid on;

    % Show the plots
    hold off;
end

%%

load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\FineTuning_tuned\TSignificant\Min_0_Max_10_Bin_8.mat")
%%
load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\FineTuning_tuned\TSignificant\Min_1_Max_7_Bin_7.mat")
%%
    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Sort narrow_EnvNormR_mean in increasing order
    [sorted_narrow_EnvNormR_mean, sortIdx_narrow] = sort(narrow_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to narrow_ABenvR_mean
    sorted_narrow_ABenvR_mean = narrow_ABenvR_mean(sortIdx_narrow);
    
    % Sort wide_EnvNormR_mean in increasing order
    [sorted_wide_EnvNormR_mean, sortIdx_wide] = sort(wide_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to wide_ABenvR_mean
    sorted_wide_ABenvR_mean = wide_ABenvR_mean(sortIdx_wide);
    
    % Plot narrow condition
    figure;
    subplot(2, 1, 1);
    plot(1:length(sorted_narrow_EnvNormR_mean), sorted_narrow_EnvNormR_mean, '-o', 'DisplayName', 'ClassicEnv (Narrow)');
    hold on;
    plot(1:length(sorted_narrow_ABenvR_mean), sorted_narrow_ABenvR_mean, '-x', 'DisplayName', 'ABenv (Narrow)');
    xlabel('Subjects (sorted by ClassicEnv values)');
    ylabel('Mean Pearson Correlation');
    title('Narrow Condition: ClassicEnv vs. ABenv (Sorted by ClassicEnv)');
    legend;
    grid on;
    
    % Plot wide condition
    subplot(2, 1, 2);
    plot(1:length(sorted_wide_EnvNormR_mean), sorted_wide_EnvNormR_mean, '-o', 'DisplayName', 'ClassicEnv (Wide)');
    hold on;
    plot(1:length(sorted_wide_ABenvR_mean), sorted_wide_ABenvR_mean, '-x', 'DisplayName', 'ABenv (Wide)');
    xlabel('Subjects (sorted by ClassicEnv values)');
    ylabel('Mean Pearson Correlation');
    title('Wide Condition: ClassicEnv vs. ABenv (Sorted by ClassicEnv)');
    legend;
    grid on;

    % Show the plots
    hold off;
%%
folder_path_plot = 'P:\AllUsers\save\TSignificant\';
file_list_plot = dir(fullfile(folder_path_plot, '*.mat'));

% Initialize array to store the narrow ABenvR_mean from each file
all_narrow_EnvNormR = zeros(1, 20);
all_narrow_ABenvR = zeros(length(file_list_plot), 20);

for i = 1:length(file_list_plot)
    % Load the file
    file_name_plot = fullfile(folder_path_plot, file_list_plot(i).name);    
    load(file_name_plot);
    disp(['Processing data from: ', file_name_plot]);

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Append narrow_ABenvR_mean to the collection
    all_narrow_EnvNormR(:) = narrow_EnvNormR_mean;
    all_narrow_ABenvR(i, :) = narrow_ABenvR_mean;

end

%%
% Initialize an array to count the number of participants for each bin range
better_performance_counts = zeros(length(file_list_plot),1); % 17 by 1 for 17 bin ranges

% Compare ABenv and ClassicEnv for each bin range
for z = 1:length(better_performance_counts)  % Loop through columns (participants)
    % Get the means for the current bin range across all files
    classic_means = all_narrow_EnvNormR;
    abenv_means = all_narrow_ABenvR(z, :);
    
    % Count how many participants have better performance with ABenv
    better_performance_counts(z) = sum(abenv_means > classic_means);
end

% Display results
disp('Bin ranges with the count of participants showing better ABenv performance than ClassicEnv:');
for z = 1:length(better_performance_counts)
    fprintf('Bin Range %d: %d participants show better performance with ABenv\n', z, better_performance_counts(z));
end

% Optionally, find the bin range with maximum participants showing better performance
[max_participants, max_bin_range] = max(better_performance_counts);
disp(['Bin range with the highest count of participants showing better performance: Bin Range ', file_list_plot(max_bin_range).name, ' with ', num2str(max_participants), ' participants.']);

%%
% Sort the better_performance_counts in descending order and get the indices
[sorted_counts, sorted_indices] = sort(better_performance_counts, 'descend');

% Find the top 10 bin ranges with the highest participant counts
top_n = 10;
top_bin_ranges = sorted_indices(1:top_n);
top_participants = sorted_counts(1:top_n);

% Display the top 10 bin ranges
for i = 1:top_n
    disp(['Rank ', num2str(i), ': Bin Range ', file_list_plot(top_bin_ranges(i)).name, ...
          ' with ', num2str(top_participants(i)), ' participants.']);
end
%% 
% Calculate the median of all narrow_ABenvR_mean values
median_narrow_ABenvR = median(all_narrow_ABenvR, 2);  % Median across subjects

disp('Median narrow ABenvR_mean across all files:');
disp(median_narrow_ABenvR);



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
    % Initialize the result to true (all normal)
    normality_result = true;
    
    % Define the features to be tested
    features = {narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean};

    for i = 1:length(features)
        % Anderson-Darling test for each feature
        [h, p] = adtest(features{i});

        % Display the results with correct interpretation
        if h == 0
            disp(['Anderson-Darling Test for Feature ' num2str(i) ': p-value = ' num2str(p) ', Normally Distributed']);
        else
            disp(['Anderson-Darling Test for Feature ' num2str(i) ': p-value = ' num2str(p) ', Not Normally Distributed']);
            % If any feature is not normally distributed, set the result to false
            normality_result = false;
        end
    end

end



%% TTest_run
function [p_ttest_Narrow, d_Narrow, p_ttest_Wide, d_Wide] = TTest_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)
    % Paired-sample t-test for Narrow condition: NormEnv vs ABEnv
    [h_Narrow, p_ttest_Narrow, ci_Narrow, stats_Narrow] = ttest(narrow_EnvNormR_mean, narrow_ABenvR_mean);
    d_Narrow = stats_Narrow.tstat / sqrt(length(narrow_EnvNormR_mean)); % Cohen's d
    if p_ttest_Narrow < 0.05
        disp(['Condition: Narrow ; Paired t-Test (NormEnv vs ABEnv)     p-value = ' num2str(p_ttest_Narrow) ' - Significant difference, Cohen''s d = ' num2str(d_Narrow)]);
    else
        disp(['Condition: Narrow ; Paired t-Test (NormEnv vs ABEnv)     p-value = ' num2str(p_ttest_Narrow) ' - No significant difference, Cohen''s d = ' num2str(d_Narrow)]);
    end

    % Paired-sample t-test for Narrow condition: NormEnv vs ABEnvNorm
    [h_Narrow_norm, p_ttest_Narrow_norm, ci_Narrow_norm, stats_Narrow_norm] = ttest(narrow_EnvNormR_mean, narrow_ABenvNormR_mean);
    d_Narrow_norm = stats_Narrow_norm.tstat / sqrt(length(narrow_EnvNormR_mean)); % Cohen's d
    if p_ttest_Narrow_norm < 0.05
        disp(['Condition: Narrow ; Paired t-Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_ttest_Narrow_norm) ' - Significant difference, Cohen''s d = ' num2str(d_Narrow_norm)]);
    else
        disp(['Condition: Narrow ; Paired t-Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_ttest_Narrow_norm) ' - No significant difference, Cohen''s d = ' num2str(d_Narrow_norm)]);
    end

    % Paired-sample t-test for Wide condition: NormEnv vs ABEnv
    [h_Wide, p_ttest_Wide, ci_Wide, stats_Wide] = ttest(wide_EnvNormR_mean, wide_ABenvR_mean);
    d_Wide = stats_Wide.tstat / sqrt(length(wide_EnvNormR_mean)); % Cohen's d
    if p_ttest_Wide < 0.05
        disp(['Condition: Wide   ; Paired t-Test (NormEnv vs ABEnv)     p-value = ' num2str(p_ttest_Wide) ' - Significant difference, Cohen''s d = ' num2str(d_Wide)]);
    else
        disp(['Condition: Wide   ; Paired t-Test (NormEnv vs ABEnv)     p-value = ' num2str(p_ttest_Wide) ' - No significant difference, Cohen''s d = ' num2str(d_Wide)]);
    end

    % Paired-sample t-test for Wide condition: NormEnv vs ABEnvNorm
    [h_Wide_norm, p_ttest_Wide_norm, ci_Wide_norm, stats_Wide_norm] = ttest(wide_EnvNormR_mean, wide_ABenvNormR_mean);
    d_Wide_norm = stats_Wide_norm.tstat / sqrt(length(wide_EnvNormR_mean)); % Cohen's d
    if p_ttest_Wide_norm < 0.05
        disp(['Condition: Wide   ; Paired t-Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_ttest_Wide_norm) ' - Significant difference, Cohen''s d = ' num2str(d_Wide_norm)]);
    else
        disp(['Condition: Wide   ; Paired t-Test (NormEnv vs ABEnvNorm) p-value = ' num2str(p_ttest_Wide_norm) ' - No significant difference, Cohen''s d = ' num2str(d_Wide_norm)]);
    end
end



%%
function aic = calculate_AIC(log_likelihood, num_parameters, sample_size)
    aic = 2 * num_parameters - 2 * log_likelihood;
end


%%
function Plot_PearsonR(sbj, narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean)


    % Sort narrow_EnvNormR_mean in increasing order
    [sorted_narrow_EnvNormR_mean, sortIdx_narrow] = sort(narrow_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to narrow_ABenvR_mean
    sorted_narrow_ABenvR_mean = narrow_ABenvR_mean(sortIdx_narrow);
    sorted_narrow_ABenvRedBinR_mean = narrow_ABenvRedBinR_mean(sortIdx_narrow);

    % Sort wide_EnvNormR_mean in increasing order
    [sorted_wide_EnvNormR_mean, sortIdx_wide] = sort(wide_EnvNormR_mean, 'ascend');
    
    % Apply the same sorting to wide_ABenvR_mean
    sorted_wide_ABenvR_mean = wide_ABenvR_mean(sortIdx_wide);
    sorted_wide_ABenvRedBinR_mean = wide_ABenvRedBinR_mean(sortIdx_wide);

    % Plot Pearson r values for narrow condition
    figure;
    subplot(2,1,1)
    hold on;
    plot(1:length(sbj), sorted_narrow_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Narrow)');
    plot(1:length(sbj), sorted_narrow_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Narrow)');
    % plot(1:length(sbj), narrow_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Narrow)');
    % plot(1:length(sbj), sorted_narrow_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins)');
    % plot(1:length(sbj), narrow_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins)');
    
    xlabel('Subjects (sorted by Narrow Standart Env values)');
    ylabel('Pearson r (mean)');
    title('Model NormEnv vs ABEnv - Narrow Condition');
    legend(location='best');
    ylim([-0.02 0.18])
    xlim([0 21])
    
    grid on;

    
    % Plot Pearson r values for wide condition
    % figure;
    subplot(2,1,2)
    hold on;
    plot(1:length(sbj), sorted_wide_EnvNormR_mean, 'bo-', 'DisplayName', 'Normal Envelope (Wide)');
    plot(1:length(sbj), sorted_wide_ABenvR_mean, 'ro-', 'DisplayName', 'AB Envelope (Wide)');
    % plot(1:length(sbj), wide_ABenvNormR_mean, 'go-', 'DisplayName', 'Normalized AB Envelope (Wide)');
    % plot(1:length(sbj), sorted_wide_ABenvRedBinR_mean, 'mo-', 'DisplayName', 'AB Envelope (Reduced Bins, Wide)');
    % plot(1:length(sbj), wide_ABenvNormRedBinR_mean, 'co-', 'DisplayName', 'Normalized AB Envelope (Reduced Bins, Wide)');
    xlabel('Subjects (sorted by Wide Standart Env values)');
    ylabel('Pearson r (mean)');
    title('Model NormEnv vs ABEnv - Wide Condition');
    legend(location='best');
    grid on;
    ylim([-0.02 0.18])
    xlim([0 21])

    % Save the plot
    fig_filename_narrow_wide = sprintf('PearsonsR');
    % save_fig(gcf, 'C:\Users\icbmadmin\Desktop\',fig_filename_narrow_wide)
    

end

%% Box_and_Ranked_Plot
% function Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)
% 
% 
%     % Define colors for the different conditions
%     color_ABenv = [0.8500, 0.3250, 0.0980]; % Red
%     color_NormEnv = [0, 0.4470, 0.7410]; % Blue
% 
%     % Create a figure for boxplots
%     figure;
% 
%     % Narrow condition participant plot
%     subplot(1, 2, 1); % Create a subplot for Narrow condition
%     hold on;
% 
%     % Plotting the data for Narrow Condition with specified colors
%     plot(1, narrow_EnvNormR_mean, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', color_NormEnv, ...
%          'MarkerFaceColor', color_NormEnv, 'DisplayName', 'NormEnv'); % NormEnv point
%     plot(2, narrow_ABenvR_mean, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', color_ABenv, ...
%          'MarkerFaceColor', color_ABenv, 'DisplayName', 'ABenv'); % ABEnv point
% 
%     % Connect points with a line for clarity
%     plot([1, 2], [narrow_EnvNormR_mean, narrow_ABenvR_mean], 'k--'); % Dashed line connecting points
% 
%     % Customize the plot
%     title('Narrow: NormEnv vs ABenv');
%     ylabel('Response Values');
%     xlim([0.5 2.5]); % Set X-axis limits
%     ylim([-0.03 0.19]);
%     set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABenv'}); % Set X-axis labels
%     grid on;
% 
%     % Wide condition participant plot
%     subplot(1, 2, 2); % Create a subplot for Wide condition
%     hold on;
% 
%     % Plotting the data for Wide Condition with specified colors
%     plot(1, wide_EnvNormR_mean, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', color_NormEnv, ...
%          'MarkerFaceColor', color_NormEnv, 'DisplayName', 'NormEnv'); % NormEnv point
%     plot(2, wide_ABenvR_mean, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', color_ABenv, ...
%          'MarkerFaceColor', color_ABenv, 'DisplayName', 'ABenv'); % ABEnv point
% 
%     % Connect points with a line for clarity
%     plot([1, 2], [wide_EnvNormR_mean, wide_ABenvR_mean], 'k--'); % Dashed line connecting points
% 
%     % Customize the plot
%     title('Wide: NormEnv vs ABenv');
%     ylabel('Response Values');
%     xlim([0.5 2.5]); % Set X-axis limits
%     ylim([-0.03 0.19]);
%     set(gca, 'XTick', [1 2], 'XTickLabel', {'NormEnv', 'ABenv'}); % Set X-axis labels
%     grid on;
% 
%     % Narrow condition
%     subplot(1, 2, 1); % Create a subplot for Narrow condition
%     h = boxplot([narrow_EnvNormR_mean, narrow_ABenvR_mean], ...
%                 'Labels', {'NormEnv', 'ABenv'});
% 
%     % Adjust line thickness for narrow condition boxplot
%     set(h, 'LineWidth', 2.5); % You can change 1.5 to a thicker value if needed
%     title('Narrow: NormEnv vs ABEnv');
%     ylabel('Response Values');
%     ylim([-0.03 0.19]);
%     grid on;
% 
%     % Wide condition
%     subplot(1, 2, 2); % Create a subplot for Wide condition
%     h2 = boxplot([wide_EnvNormR_mean, wide_ABenvR_mean], ...
%                 'Labels', {'NormEnv', 'ABenv'});
% 
%     % Adjust line thickness for wide condition boxplot
%     set(h2, 'LineWidth', 2.5); % Again, adjust to make it thicker if necessary
%     title('Wide: NormEnv vs ABEnv');
%     ylabel('Response Values');
%     ylim([-0.03 0.19]);
%     grid on;
% 
% end

function Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean)

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
    file_name_plot = fullfile(folder_path, file_list(i).name);
    load(file_name_plot);
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


%% Wide
OT_setup
%%

folder_path_plot = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoopAll\';
file_list_plot = dir(fullfile(folder_path_plot, '*.mat'));

% Initialize array to store the wide ABenvR_mean from each file
all_wide_EnvNormR = zeros(1, 20);
all_wide_ABenvR = zeros(length(file_list_plot), 20);

for i = 1:length(file_list_plot)
    % Load the file
    file_name_plot = fullfile(folder_path_plot, file_list_plot(i).name);    
    load(file_name_plot);
    disp(['Processing data from: ', file_name_plot]);

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Append widew_ABenvR_mean to the collection
    all_wide_EnvNormR(:) = wide_EnvNormR_mean;
    all_wide_ABenvR(i, :) = wide_ABenvR_mean;

end

% Initialize an array to count the number of participants for each bin range
better_performance_counts_wide = zeros(length(file_list_plot),1); % 17 by 1 for 17 bin ranges

% Compare ABenv and ClassicEnv for each bin range
for z = 1:length(better_performance_counts_wide)  % Loop through columns (participants)
    % Get the means for the current bin range across all files
    classic_means = all_wide_EnvNormR;
    abenv_means = all_wide_ABenvR(z, :);
    
    % Count how many participants have better performance with ABenv
    better_performance_counts_wide(z) = sum(abenv_means > classic_means);
end

% Display results
disp('Bin ranges with the count of participants showing better ABenv performance than ClassicEnv:');
for z = 1:length(better_performance_counts_wide)
    fprintf('Bin Range %d: %d participants show better performance with ABenv\n', z, better_performance_counts_wide(z));
end

% Optionally, find the bin range with maximum participants showing better performance
[max_participants_wide, max_bin_range_wide] = max(better_performance_counts_wide);
disp(['Bin range with the highest count of participants showing better performance: Bin Range ', num2str(max_bin_range_wide), ' with ', num2str(max_participants_wide), ' participants.']);
%%
%% Narrow condition analysis
folder_path_plot = 'P:\AllUsers\save\TSignificant\';
file_list_plot = dir(fullfile(folder_path_plot, '*.mat'));

% Initialize arrays to store the narrow ABenvR_mean and EnvNormR_mean from each file
all_narrow_EnvNormR = zeros(1, 20); % Change 20 to the actual number of bins if different
all_narrow_ABenvR = zeros(length(file_list_plot), 20);

for i = 1:length(file_list_plot)
    % Load the file
    file_name_plot = fullfile(folder_path_plot, file_list_plot(i).name);    
    load(file_name_plot);
    disp(['Processing data from: ', file_name_plot]);

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Store narrow data
    all_narrow_EnvNormR(:) = narrow_EnvNormR_mean;
    all_narrow_ABenvR(i, :) = narrow_ABenvR_mean;
end

% Initialize an array to count the number of participants for each bin range (narrow condition)
better_performance_counts_narrow = zeros(length(file_list_plot), 1);

% Compare ABenv and ClassicEnv for each bin range in narrow condition
for z = 1:length(better_performance_counts_narrow)
    classic_means = all_narrow_EnvNormR;
    abenv_means = all_narrow_ABenvR(z, :);
    
    % Count how many participants have better performance with ABenv
    better_performance_counts_narrow(z) = sum(abenv_means > classic_means);
end

% Display results for narrow condition
disp('Narrow condition: Bin ranges with count of participants showing better ABenv performance than ClassicEnv:');
for z = 1:length(better_performance_counts_narrow)
    fprintf('Bin Range %d: %d participants show better performance with ABenv\n', z, better_performance_counts_narrow(z));
end

% Sort and display the top 10 bin ranges for narrow condition
[sorted_counts_narrow, sorted_indices_narrow] = sort(better_performance_counts_narrow, 'descend');
top_n = 10;
top_bin_ranges_narrow = sorted_indices_narrow(1:top_n);
top_participants_narrow = sorted_counts_narrow(1:top_n);

disp('Top 10 bin ranges for narrow condition:');
for i = 1:top_n
    disp(['Rank ', num2str(i), ': Bin Range ', file_list_plot(top_bin_ranges_narrow(i)).name, ...
          ' with ', num2str(top_participants_narrow(i)), ' participants.']);
end

%% Wide condition analysis
% Initialize arrays to store the wide ABenvR_mean and EnvNormR_mean from each file
all_wide_EnvNormR = zeros(1, 20); % Change 20 to the actual number of bins if different
all_wide_ABenvR = zeros(length(file_list_plot), 20);

for i = 1:length(file_list_plot)
    % Load the file
    file_name_plot = fullfile(folder_path_plot, file_list_plot(i).name);    
    load(file_name_plot);
    disp(['Processing data from: ', file_name_plot]);

    % Perform the analysis
    [narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, ...
        narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);
    
    % Store narrow data
    all_wide_EnvNormR(:) = wide_EnvNormR_mean;
    all_wide_ABenvR(i, :) = wide_ABenvR_mean;
end

%%
% Initialize an array to count the number of participants for each bin range (wide condition)
better_performance_counts_wide = zeros(length(file_list_plot), 1);

% Compare ABenv and ClassicEnv for each bin range in wide condition
for z = 1:length(better_performance_counts_wide)
    classic_means_wide = all_wide_EnvNormR;
    abenv_means_wide = all_wide_ABenvR(z, :);
    
    % Count how many participants have better performance with ABenv
    better_performance_counts_wide(z) = sum(abenv_means_wide > classic_means_wide);
end

% Display results for wide condition
disp('Wide condition: Bin ranges with count of participants showing better ABenv performance than ClassicEnv:');
for z = 1:length(better_performance_counts_wide)
    fprintf('Bin Range %d: %d participants show better performance with ABenv\n', z, better_performance_counts_wide(z));
end

% Sort and display the top 10 bin ranges for wide condition
[sorted_counts_wide, sorted_indices_wide] = sort(better_performance_counts_wide, 'descend');
top_bin_ranges_wide = sorted_indices_wide(1:top_n);
top_participants_wide = sorted_counts_wide(1:top_n);

disp('Top 10 bin ranges for wide condition:');
for i = 1:top_n
    disp(['Rank ', num2str(i), ': Bin Range ', file_list_plot(top_bin_ranges_wide(i)).name, ...
          ' with ', num2str(top_participants_wide(i)), ' participants.']);
end


%%
% clear; 
clc; OT_setup
%%
[narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, ...
    narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean] = extraction_pearson(sbj, task, name_struct_new);

NormalityTest(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean,  wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);

Plot_PearsonR(sbj, narrow_EnvNormR_mean, wide_EnvNormR_mean, narrow_ABenvR_mean, wide_ABenvR_mean, narrow_ABenvNormR_mean, wide_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_ABenvRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean);

% % 3 Features - Without Bin Reduction
[p_ttest_Narrow, d_Narrow, p_ttest_Wide, d_Wide] = TTest_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);
%%
Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormR_mean);

%% 5 Features - With Bin Reduction
% TTest_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvRedBinR_mean)
Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvNormR_mean, narrow_ABenvRedBinR_mean, wide_EnvNormR_mean, wide_ABenvNormR_mean, wide_ABenvRedBinR_mean)
%% 5 Features - With norm Bin Reduction
% Wilcoxon_run(narrow_EnvNormR_mean, narrow_ABenvR_mean, narrow_ABenvNormRedBinR_mean, wide_EnvNormR_mean, wide_ABenvR_mean, wide_ABenvNormRedBinR_mean)
Box_and_Ranked_Plot(narrow_EnvNormR_mean, narrow_ABenvNormRedBinR_mean, narrow_ABenvNormRedBinR_mean, wide_EnvNormR_mean, wide_ABenvNormRedBinR_mean, wide_ABenvNormRedBinR_mean)
