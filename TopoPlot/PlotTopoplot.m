function [matrix_StdEnv_Narrow, matrix_ABenv_Narrow, matrix_StdEnv_Wide, matrix_ABenv_Wide] =PlotTopoplot(StatsParticipantTask, chanlocsData, sbj)
    
    

    % Narrow
    matrix_StdEnv_Narrow = zeros(20, 22);
    matrix_ABenv_Narrow = zeros(20, 22);
    % Wide
    matrix_StdEnv_Wide = zeros(20, 22);
    matrix_ABenv_Wide = zeros(20, 22);
    
    for s = 1:length(sbj)
        % Narrow
        matrix_StdEnv_Narrow(s, :) = StatsParticipantTask(s, 1).EnvNormStats.r;
        matrix_ABenv_Narrow(s, :) = StatsParticipantTask(s, 1).ABenvStats.r;
        
        % Wide
        matrix_StdEnv_Wide(s, :) = StatsParticipantTask(s, 2).EnvNormStats.r;
        matrix_ABenv_Wide(s, :) = StatsParticipantTask(s, 2).ABenvStats.r;
    end
    
    avg_matrix_StdEnv_Narrow = mean(matrix_StdEnv_Narrow, 1);
    avg_matrix_ABenv_Narrow = mean(matrix_ABenv_Narrow, 1);
    avg_matrix_StdEnv_Wide = mean(matrix_StdEnv_Wide, 1);
    avg_matrix_ABenv_Wide = mean(matrix_ABenv_Wide, 1);
    
    %%% Plot
    
    % Calculate the global maximum for Narrow and Wide separately
    global_max = max([max(avg_matrix_StdEnv_Narrow), ...
                       max(avg_matrix_ABenv_Narrow), ...
                       max(avg_matrix_StdEnv_Wide), ...
                       max(avg_matrix_ABenv_Wide)]);
    
    % global_min = max([min(avg_matrix_StdEnv_Narrow), min(avg_matrix_ABenv_Narrow), ...
    % min(avg_matrix_StdEnv_Wide), min(avg_matrix_ABenv_Wide)]);
    
    color_lim_global = [0, global_max];  % Limits for Narrow plots
    
    % figure;
    
    % Plot Narrow
    subplot(2,2,1); title(sprintf('Topoplot (NormEnv), Narrow' ));  % Title based on the feature name
    topoplot(avg_matrix_StdEnv_Narrow, chanlocsData);  
    c = colorbar('location', 'westoutside'); c.Label.String = "Pearson's r"; clim(color_lim_global);
set(gca, 'FontSize', 15); % Adjust the font size for tick labels

    subplot(2,2,2); title(sprintf('Topoplot (ABenv), Narrow' ));  % Title based on the feature name
    topoplot(avg_matrix_ABenv_Narrow, chanlocsData);
    c = colorbar('location', 'westoutside'); c.Label.String = "Pearson's r"; clim(color_lim_global);
 set(gca, 'FontSize', 15); % Adjust the font size for tick labels

    % Plot Wide
    subplot(2,2,3); title(sprintf('Topoplot (NormEnv), Wide' ));  % Title based on the feature name
    topoplot(avg_matrix_StdEnv_Wide, chanlocsData);  
    c = colorbar('location', 'westoutside'); c.Label.String = "Pearson's r"; clim(color_lim_global);
set(gca, 'FontSize', 15); % Adjust the font size for tick labels

    subplot(2,2,4); title(sprintf('Topoplot (ABenv), Wide' ));  % Title based on the feature name
    topoplot(avg_matrix_ABenv_Wide, chanlocsData);  
    c = colorbar('location', 'westoutside'); c.Label.String = "Pearson's r"; clim(color_lim_global);
set(gca, 'FontSize', 15); % Adjust the font size for tick labels



    %% Print name and values
    % Sort and get top 5 channels for Standard and AB Envelopes
    [sorted_StdEnv_Narrow, idx_StdEnv_Narrow] = sort(avg_matrix_StdEnv_Narrow, 'descend');
    [sorted_ABenv_Narrow, idx_ABenv_Narrow] = sort(avg_matrix_ABenv_Narrow, 'descend');
    [sorted_StdEnv_Wide, idx_StdEnv_Wide] = sort(avg_matrix_StdEnv_Wide, 'descend');
    [sorted_ABenv_Wide, idx_ABenv_Wide] = sort(avg_matrix_ABenv_Wide, 'descend');
    
    % Get the top 5 channels for each condition
    top_5_channels_StdEnv_Narrow = {chanlocsData(idx_StdEnv_Narrow(1:5)).labels};
    top_5_channels_ABenv_Narrow = {chanlocsData(idx_ABenv_Narrow(1:5)).labels};
    top_5_channels_StdEnv_Wide = {chanlocsData(idx_StdEnv_Wide(1:5)).labels};
    top_5_channels_ABenv_Wide = {chanlocsData(idx_ABenv_Wide(1:5)).labels};
    
    % Display the top 5 channels and their values
    fprintf('Top 5 Channels (Env, Narrow):\n');
    for i = 1:5
        fprintf('Channel %s has an average value of %.4f\n', top_5_channels_StdEnv_Narrow{i}, sorted_StdEnv_Narrow(i));
    end
    
    fprintf('\nTop 5 Channels (ABenv, Narrow):\n');
    for i = 1:5
        fprintf('Channel %s has an average value of %.4f\n', top_5_channels_ABenv_Narrow{i}, sorted_ABenv_Narrow(i));
    end
    
    fprintf('\nTop 5 Channels (Env, Wide):\n');
    for i = 1:5
        fprintf('Channel %s has an average value of %.4f\n', top_5_channels_StdEnv_Wide{i}, sorted_StdEnv_Wide(i));
    end
    
    fprintf('\nTop 5 Channels (ABenv, Wide):\n');
    for i = 1:5
        fprintf('Channel %s has an average value of %.4f\n', top_5_channels_ABenv_Wide{i}, sorted_ABenv_Wide(i));
    end

end