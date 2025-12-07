% Define parameter ranges
min_bin_values = [0, 10, 20, 30, 40, 50, 60, 70];
max_bin_values = [6, 10, 20, 30, 40, 50, 60, 70, 80];
num_bins_values = [4, 5, 6, 8, 12, 16];

% Predefine folder paths
ABworsefolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\3.Significant Diff\5.ABworse\';
ABBetterfolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\3.Significant Diff\4.ABbetter_narrow\'; 
Nosignificantfolder_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop\1.No Significant Diff';

% Load configurations from files
ABworse_files = dir(fullfile(ABworsefolder_path, '*.mat'));
ABBetter_files = dir(fullfile(ABBetterfolder_path, '*.mat'));
Nosignificant_files = dir(fullfile(Nosignificantfolder_path, '*.mat'));

% Function to extract min_bin, max_bin, and num_bins from filename
extract_params = @(filename) cellfun(@str2double, regexp(filename, '\d+', 'match'));

% Initialize containers for configurations
ab_worse_configs = [];
ab_better_configs = [];
ab_nosignificance_configs = [];

% Extract parameters for each condition
for i = 1:length(ABworse_files)
    ab_worse_configs = [ab_worse_configs; extract_params(ABworse_files(i).name)];
end
for i = 1:length(ABBetter_files)
    ab_better_configs = [ab_better_configs; extract_params(ABBetter_files(i).name)];
end
for i = 1:length(Nosignificant_files)
    ab_nosignificance_configs = [ab_nosignificance_configs; extract_params(Nosignificant_files(i).name)];
end

% Prepare subplots for each num_bins value
figure;
colormap jet;

for i = 1:length(num_bins_values)
    num_bins = num_bins_values(i);

    % Initialize matrix to store binary significance for each min-max bin pair
    heatmap_data = NaN(length(min_bin_values), length(max_bin_values));

    % Populate heatmap data based on configurations
    for j = 1:size(ab_worse_configs, 1)
        if ab_worse_configs(j, 3) == num_bins
            min_idx = find(min_bin_values == ab_worse_configs(j, 1));
            max_idx = find(max_bin_values == ab_worse_configs(j, 2));
            heatmap_data(min_idx, max_idx) = -1; % AB Worse in red
        end
    end
    for j = 1:size(ab_better_configs, 1)
        if ab_better_configs(j, 3) == num_bins
            min_idx = find(min_bin_values == ab_better_configs(j, 1));
            max_idx = find(max_bin_values == ab_better_configs(j, 2));
            heatmap_data(min_idx, max_idx) = 1; % AB Better in green
        end
    end
    for j = 1:size(ab_nosignificance_configs, 1)
        if ab_nosignificance_configs(j, 3) == num_bins
            min_idx = find(min_bin_values == ab_nosignificance_configs(j, 1));
            max_idx = find(max_bin_values == ab_nosignificance_configs(j, 2));
            heatmap_data(min_idx, max_idx) = 0; % No significance in yellow
        end
    end

    % Plot heatmap for current num_bins
    subplot(2, ceil(length(num_bins_values) / 2), i); % Arrange in 2 rows
    imagesc(heatmap_data, [-1 1]); % Color range from -1 to 1
    title(['Num Bins = ', num2str(num_bins)]);
    xlabel('Max Bin');
    ylabel('Min Bin');
    xticks(1:length(max_bin_values));
    xticklabels(max_bin_values);
    yticks(1:length(min_bin_values));
    yticklabels(min_bin_values);
    colorbar;
end

sgtitle('Significance Colormaps for Each Number of Bins');


