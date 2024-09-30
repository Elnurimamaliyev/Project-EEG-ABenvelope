function PlotTopoplot(StatsParticipantTask, chanlocsData, sbj)
    
    

    % Narrow
    matrix_StdEnv_Narrow = zeros(20, 22);
    matrix_ABenv_Narrow = zeros(20, 22);
    % Wide
    matrix_StdEnv_Wide = zeros(20, 22);
    matrix_ABenv_Wide = zeros(20, 22);
    
    for s = 1:length(sbj)
        % Narrow
        matrix_StdEnv_Narrow(s, :) = StatsParticipantTask(s, 1).NormEnvStats.r;
        matrix_ABenv_Narrow(s, :) = StatsParticipantTask(s, 1).ABenvStats.r;
        
        % Wide
        matrix_StdEnv_Wide(s, :) = StatsParticipantTask(s, 2).NormEnvStats.r;
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
    
    figure;
    
    % Plot Narrow
    subplot(2,2,1); title(sprintf('Topoplot (NormEnv), Narrow' ));  % Title based on the feature name
    topoplot(avg_matrix_StdEnv_Narrow, chanlocsData);  colorbar();clim(color_lim_global);
    
    
    subplot(2,2,2); title(sprintf('Topoplot (ABenv), Narrow' ));  % Title based on the feature name
    topoplot(avg_matrix_ABenv_Narrow, chanlocsData);  colorbar();clim(color_lim_global);
    
    % Plot Wide
    subplot(2,2,3); title(sprintf('Topoplot (NormEnv), Wide' ));  % Title based on the feature name
    topoplot(avg_matrix_StdEnv_Wide, chanlocsData);  colorbar();clim(color_lim_global);
    
    subplot(2,2,4); title(sprintf('Topoplot (ABenv), Wide' ));  % Title based on the feature name
    topoplot(avg_matrix_ABenv_Wide, chanlocsData);  colorbar(); clim(color_lim_global);

end