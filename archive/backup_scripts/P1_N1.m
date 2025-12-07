% StatsParticipantTask_27_10
clear;clc;
%% 
OT_setup
load("StatsParticipantTask_27_10.mat")

StatsParticipantTask_N1_P2 = struct();
%% Bins
step_size=0.98/8; binEdges_dB = linspace(0,0.98,8); numBins = length(binEdges_dB); 
binEdges_amp = 10.^(binEdges_dB / 20); normbinEdges_amp = normalize(binEdges_amp,2,'range');

%%
OT_setup
load("StatsParticipantTask_27_10.mat")

% Initialize arrays to store the mean weights for all subjects
weights_all_subj_Narrow = zeros(length(sbj), 9, 51); % 20 subjects, 9 bins, 51 samples
weights_all_subj_Wide = zeros(length(sbj), 9, 51); % 20 subjects, 9 bins, 51 samples

% Loop through all subjects and tasks
for s = 1:length(sbj)
    % Process each feature
    Modelfeatures = {'ABenvNormModel'};
    
    for feature_idx = 1:length(Modelfeatures)
        Model_Narrow = StatsParticipantTask_27_10(s, 1).(Modelfeatures{feature_idx});
        Model_Wide = StatsParticipantTask_27_10(s, 2).(Modelfeatures{feature_idx});
        
        % Extract time vector and mean weights across channels for each bin
        t_vector = Model_Narrow.t;
        weights_Narrow = Model_Narrow.w; % Dimensions (9, nSamples, nChannels)
        mean_weights_bins_Narrow = squeeze(mean(weights_Narrow, 3)); % Average over channels, resulting in (9, nSamples)

        weights_Wide = Model_Wide.w; % Dimensions (9 bins, ntime, nChannels)
        mean_weights_bins_Wide = squeeze(mean(weights_Wide, 3)); % Average over channels, resulting in (9, nSamples)
        % Store the mean weights for this subject in the structure
        weights_all_subj_Narrow(s, :, :) = mean_weights_bins_Narrow; % Store for this subject (9 bins x 51 samples)
        weights_all_subj_Wide(s, :, :) = mean_weights_bins_Wide; % Store for this subject (9 bins x 51 samples)

        % figure;
        % % Plot for Narrow condition
        % subplot(2,1,1)
        % hold on;
        % colors = lines(9); % Generate distinct colors for each bin (9 bins now)
        % step_size = 1/8; % Adjust the step size for visualization
        % 
        % for bin_idx = 1:8
        %     plot(t_vector_Narrow, mean_weights_bins_Narrow(bin_idx, :) * step_size + bin_idx, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
        %          'DisplayName', sprintf('Bin %d', bin_idx));
        % end
        % title('Averaged Mean Weights for Each Bin (Narrow Condition)');
        % xlabel('Time (ms)');
        % ylabel('Mean Weight (a.u.)');
        % legend show;
        % hold off;
        % 
        % % Plot for Wide condition
        % subplot(2,1,2)
        % hold on;
        % for bin_idx = 1:8
        %     plot(t_vector, mean_weights_bins_Wide(bin_idx, :) * step_size + bin_idx, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
        %          'DisplayName', sprintf('Bin %d', bin_idx));
        % end
        % title('Averaged Mean Weights for Each Bin (Wide Condition)');
        % xlabel('Time (ms)');
        % ylabel('Mean Weight (a.u.)');
        % legend show;
        % hold off;

    end
end

% Exclude subjects 2, 5, and 18 for group mean calculation
exclude_subjects = [2, 5, 18]; valid_subjects = setdiff(1:length(sbj), exclude_subjects);  % Find subjects not in the exclude list
% Calculate the average mean weights across the valid subjects for each task
mean_weights_Narrow_avg = squeeze(mean(weights_all_subj_Narrow(valid_subjects, :, :), 1)); % Averaging across valid subjects (axis 1)
mean_weights_Wide_avg = squeeze(mean(weights_all_subj_Wide(valid_subjects, :, :), 1)); % Averaging across valid subjects (axis 1)
AVG_Weights = (mean_weights_Narrow_avg + mean_weights_Wide_avg)/2;
%% 
% Plot for Group and Condition average with offsets for each bin and peak markers
P2start=0;
P2stop=170;
N1start=30;
N1stop=150;

% Colors
blue = [0, 0.4470, 0.7410]; red = [0.6350 0.0780 0.1840];

%%
figure;
subplot(4,4, [1,2,5,6])
% subplot(2,2, 1)
hold on;
colors = repmat([0.5 0 0.5], 8, 1); % Define a color for each bin

% Min-max normalization across the entire matrix
min_val = min(AVG_Weights(:));
max_val = max(AVG_Weights(:));
AVG_Weights_norm = (AVG_Weights - min_val) / (max_val - min_val);

% Define the time window indices
[~, P2_start_idx] = min(abs(t_vector - P2start));
[~, P2_stop_idx] = min(abs(t_vector - P2stop));
[~, N1_start_idx] = min(abs(t_vector - N1start));
[~, N1_stop_idx] = min(abs(t_vector - N1stop));

N1_peak_values = zeros(8,1);
N1_peak_times = zeros(8,1);
P2_peak_values= zeros(8,1);
P2_peak_times= zeros(8,1);

for bin_idx = 1:8
    % Calculate Original peaks 
    Bin_weight = AVG_Weights(bin_idx, :);
    [P2_peak_ORG, P2_idx_plot_ORG] = max(Bin_weight(P2_start_idx:P2_stop_idx));
    P2_time_ORG = t_vector(P2_start_idx + P2_idx_plot_ORG - 1); 

    [N1_peak_ORG, N1_idx_ORG] = min(Bin_weight(N1_start_idx:N1_stop_idx));
    N1_time_ORG = t_vector(N1_start_idx + N1_idx_ORG - 1); 

    % Save Original Peak values
    N1_peak_values(bin_idx) = N1_peak_ORG;
    N1_peak_times(bin_idx) = N1_time_ORG;
    P2_peak_values(bin_idx) = P2_peak_ORG;
    P2_peak_times(bin_idx) = P2_time_ORG;

    %%% Plot
        % Calculation 
        plot_AVG_weights = AVG_Weights_norm(bin_idx, :);
        offset = (bin_idx - 1) * step_size - 0.5; % Adjust the offset to separate lines
        % Find N1 peak 
        [N1_peak, N1_idx] = min(plot_AVG_weights(N1_start_idx:N1_stop_idx));
        N1_time = t_vector(N1_start_idx + N1_idx - 1); 
    
        % Find P2 peak 
        [P2_peak_plot, P2_idx_plot] = max(plot_AVG_weights(P2_start_idx:P2_stop_idx));
        P2_time = t_vector(P2_start_idx + P2_idx_plot - 1); 

        % Plot weights (norm) with offset
        plot(t_vector, plot_AVG_weights + offset, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
             'DisplayName', sprintf('Bin %d', bin_idx));
        % N1 Plot
        plot(N1_time, N1_peak + offset, 'o', 'MarkerSize', 5, 'LineWidth', 1.5, 'MarkerEdgeColor', 'b', ...
             'DisplayName', sprintf('N1 Peak Bin %d', bin_idx));
        % P2 Peak
        plot(P2_time, P2_peak_plot + offset, 'o', 'MarkerSize', 5, 'LineWidth', 1.5, 'MarkerEdgeColor', 'r', ...
             'DisplayName', sprintf('P2 Peak Bin %d', bin_idx));
end
legend('Weights','N1 Peak','P2 Peak')

title('Group & Condition Average Weights'); 
xlabel('Time lag (ms)'); ylabel('Amplitude Bins (a.u.)');
xlim([-100, 400]); hold off;
box off

% %%
% Subplot 2: Image plot of group average AB envelope TRF
window = (2:50);

subplot(4,4,[3,4,7,8]);
% subplot(2,2,2);
AVG_Weights_updownnorm = AVG_Weights/max(AVG_Weights(:));
imagesc(t_vector(window), normbinEdges_amp(1:8), AVG_Weights_updownnorm(1:8,window)); % Replace '1:8' if you have specific bin labels or values
title('Weights Image');
xlabel('Time lag (ms)');
ylabel('Amplitude Bins (a.u.)');
xlim([-100, 400]);
set(gca, 'YDir', 'normal');  % Ensure Y-axis is in the natural direction
box off

colormap('jet'); % Use 'hot' or any other colormap you prefer
cb = colorbar("Position", [0.9131 0.5835 0.0244, 0.3421]);  % Places the color bar below the plot
cb.Label.String = 'a.u.';     % Set the color bar label
cb.Label.Rotation = 270;
cb.Label.Position = [2.45,0];


% Subplot 3
% subplot(2,2,3)
% title('N1 and P2 Peak Latencies'); 
% %%
% subplot(4,4,[9,10])
% % hold on
% plot(N1_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', blue,'DisplayName','N1'); % Use 'Color' to set lighter blue
% % plot(P2_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', red, 'DisplayName','P2'); % Use 'Color' to set lighter blue
% title('N1 and P2 Peak Latencies', FontSize=10); 
% 
% % title('N1 Peak Latencies'); 
% % xlabel('Time lag (ms)'); 
% ylabel('Amplitude Bins (a.u.)');
% ylim([-0.1, 1.1]); 
% % ylim([-0.05, 1.05]);
% xlim([05, 105]);
% 
% grid on; hold off;box off;
% legend(Location="best")
% 
% subplot(4,4,[13,14])
% plot(P2_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', red, 'DisplayName','P2'); % Use 'Color' to set lighter blue
% % title('P2 Peak Latencies'); 
% xlabel('Time lag (ms)');
% ylabel('Amplitude Bins (a.u.)');
% ylim([-0.1, 1.1]); 
% xlim([105, 205]);
% % title('N1 and P2 Peak Latencies'); 
% % ylabel('Amplitude Bins (a.u.)');
% grid on; hold off;box off;
% legend(Location="best")


%% Plot N1 and P2 Peak Latencies and Fit Lines

% N1 Peak Latencies
subplot(4,4,[9,10])
hold on
% Plot N1 peak latencies
plot(N1_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', blue, 'DisplayName','N1'); 

% Fit a linear model for N1
[p_n1, S_n1] = polyfit(normbinEdges_amp, N1_peak_times, 1);  % Amplitude vs N1 peak times
yfit_n1 = polyval(p_n1, normbinEdges_amp);
plot(yfit_n1, normbinEdges_amp(1:8), '--', 'LineWidth', 1.5, 'Color', 'k', 'DisplayName','N1 Fit');

% Title and labels for N1
title('N1 and P2 Peak Latencies', 'FontSize', 10); 
xlabel('Time lag (ms)');
ylabel('Amplitude Bins (a.u.)');
ylim([-0.1, 1.1]); 
xlim([5, 105]);
legend('Location','best');

% Calculate R² for the N1 fit
yresid_n1 = N1_peak_times' - yfit_n1;  % Residuals
SSresid_n1 = sum(yresid_n1.^2);  % Sum of squared residuals
SStotal_n1 = (length(N1_peak_times) - 1) * var(N1_peak_times);  % Total sum of squares
R2_n1 = 1 - (SSresid_n1 / SStotal_n1);  % R² value

% Slope and change per bin for N1
slope_n1 = p_n1(1);  % Slope from the linear fit
change_per_bin_n1 = slope_n1 / 8;  % Dividing the slope by the number of bins (8 bins)

% Calculate p-value for N1 using regress (requires the full model matrix)
X_n1 = [ones(length(normbinEdges_amp), 1), normbinEdges_amp(:)]; % Design matrix (with intercept term)
[b_n1, bint_n1, r_n1, rint_n1, stats_n1] = regress(N1_peak_times(:), X_n1);  % Perform linear regression
p_value_n1 = stats_n1(3);  % p-value from the regression statistics

% Display the results for N1
fprintf('For N1: A line was then fit to the data (R² = %.2f, p = %.3f), which showed that the N1 peak latency increases by %.3f ms with every unit decrease in amplitude bin.\n', R2_n1, p_value_n1, change_per_bin_n1);


% P2 Peak Latencies
subplot(4,4,[13,14])
hold on
% Plot P2 peak latencies
plot(P2_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', red, 'DisplayName','P2');

% Fit a linear model for P2
[p_p2, S_p2] = polyfit(normbinEdges_amp, P2_peak_times, 1);  % Amplitude vs P2 peak times
yfit_p2 = polyval(p_p2, normbinEdges_amp);
plot(yfit_p2, normbinEdges_amp(1:8), '--', 'LineWidth', 1.5, 'Color', 'k', 'DisplayName','P2 Fit');

% Title and labels for P2
xlabel('Time lag (ms)');
ylabel('Amplitude Bins (a.u.)');
ylim([-0.1, 1.1]); 
xlim([105, 205]);

% Calculate R² for the P2 fit
yresid_p2 = P2_peak_times' - yfit_p2;  % Residuals
SSresid_p2 = sum(yresid_p2.^2);  % Sum of squared residuals
SStotal_p2 = (length(P2_peak_times) - 1) * var(P2_peak_times);  % Total sum of squares
R2_p2 = 1 - (SSresid_p2 / SStotal_p2);  % R² value

% Slope and change per bin for P2
slope_p2 = p_p2(1);  % Slope from the linear fit
change_per_bin_p2 = slope_p2 / 8;  % Dividing the slope by the number of bins (8 bins)

% Calculate p-value for P2 using regress (requires the full model matrix)
X_p2 = [ones(length(normbinEdges_amp), 1), normbinEdges_amp(:)]; % Design matrix (with intercept term)
[b_p2, bint_p2, r_p2, rint_p2, stats_p2] = regress(P2_peak_times(:), X_p2);  % Perform linear regression
p_value_p2 = stats_p2(3);  % p-value from the regression statistics

% Display the results for P2
fprintf('For P2: A line was then fit to the data (R² = %.2f, p = %.3f), which showed that the P2 peak latency increases by %.4f ms with every unit decrease in amplitude bin.\n', R2_p2, p_value_p2, change_per_bin_p2);

% Final grid and legend settings
grid on; 
hold off;
box off;
legend('Location','best');

%%
% Calculation
Diff_P2_N1_vals = P2_peak_values - N1_peak_values;

% Subplot 4
subplot(4,4,[11,12,15,16]);

% Ensure that amplitudes and Diff_P2_N1_vals have matching lengths
amplitudes = normbinEdges_amp(1:8);  % Amplitude bins (x)
Magnetude_diff = Diff_P2_N1_vals';   % Difference in magnitude (y)

% Plot the difference between P2 and N1 (using normalized values)
plot(Magnetude_diff, amplitudes,  'o-', 'LineWidth', 1.5, 'Color', blue);  % Plot normalized differences (y) vs amplitudes (x)
hold on;

% Fit a linear model (degree 1) to the normalized data
[p_diff, S_diff] = polyfit(amplitudes, Magnetude_diff, 1);  % Linear fit (amplitudes vs normalized differences)
yfit_diff = polyval(p_diff, amplitudes);  % Calculate fitted values based on amplitudes

% Plot the linear fit
plot(yfit_diff,amplitudes, '--', 'LineWidth', 1.5, 'Color', 'k', 'DisplayName','Fit'); 

% Title and labels
title('P2 - N1 Difference');
ylabel('Amplitude bins (a.u.)');
xlabel('Magnitude (a.u.)');
legend('Location', 'best');
grid on;

% Calculate R² for the fit
yresid_diff = Magnetude_diff - yfit_diff;  % Residuals
SSresid_diff = sum(yresid_diff.^2);  % Sum of squared residuals
SStotal_diff = (length(Magnetude_diff) - 1) * var(Magnetude_diff);  % Total sum of squares
R2_diff = 1 - (SSresid_diff / SStotal_diff);  % R² value

% Slope (p_diff(1)) and intercept
slope_diff = p_diff(1);  % Slope from the linear fit
intercept_diff = p_diff(2);  % Intercept from the linear fit

% Calculate the "change" as (last fitted value - first fitted value) / number of bins
first_fitted_value = yfit_diff(1);  % First fitted value (for amplitude = first bin)
last_fitted_value = yfit_diff(end);  % Last fitted value (for amplitude = last bin)
change = (last_fitted_value - first_fitted_value) / (length(amplitudes) - 1);  % Divide by number of bins

% Perform regression to calculate p-value
X = [ones(length(Magnetude_diff), 1), amplitudes(:)];  % Ensure amplitudes is a column vector
[b, bint, r, rint, stats] = regress(Magnetude_diff', X);  % Linear regression
p_value = stats(3);  % p-value for the slope

% Display the results
fprintf('For P2 - N1 Difference: A line was fit to the data (R² = %.2f, p = %.3f). The slope is %.4f, indicating a %.5f change in normalized magnitude per unit amplitude difference.\n', R2_diff, p_value, slope_diff, change);

% Final grid and legend settings
box off;
hold off;
legend('Location', 'best');
























%%
xlim([0 100])
ylim([-0.05 1.05])
%%

%%
hAxes = findobj(gcf,"Type","axes")
hAxes(3).Position = [0.1144,0.0720,0.1566,0.3768]
hAxes(2).Position = [0.3011,0.0703,0.1566,0.3768]
hAxes(5).Position = [0.1118,0.5482,0.3628,0.3768]
hAxes(4).Position = [0.5422,0.5482,0.3628,0.3768]
hAxes(1).Position = [0.5422,0.0687,0.3628,0.3768]
%%
figure;
plot(N1_peak_times, normbinEdges_amp(1:8), 'o-', 'MarkerSize', 5,'LineWidth', 1.5, 'Color', blue,'DisplayName','N1'); % Use 'Color' to set lighter blue
%% 
























figure;
% Display the TRF as an image plot (AVG_Weights should be the TRF data matrix)
window = (2:50);
AVG_Weights_updownnorm = AVG_Weights/max(AVG_Weights(:));
imagesc(t_vector(window), normbinEdges_amp(1:8), AVG_Weights_updownnorm(1:8,window)); % Replace '1:8' if you have specific bin labels or values

% Adjust color scale, axis labels, and colorbar
colormap('jet'); % Use 'hot' or any other colormap you prefer
% 
cb = colorbar('eastoutside');  % Places the color bar below the plot
cb.Label.String = 'a.u.';     % Set the color bar label
cb.Label.Rotation = 270;      % Rotate label for readability (optional)
% cb.Label.VerticalAlignment = 'top';  % Position adjustment (optional)
title('Group Average AB Envelope TRF');
xlabel('Time lag (ms)');
ylabel('Amplitude Bins');


%% 
% Plot for Narrow condition
figure;
% subplot(1,2,1)
hold on;
% colors = lines(8); % Generate distinct colors for each bin
colors = repmat([0.5 0 0.5], 8, 1); % Generate distinct colors for each bin
step_size = 1/8; % Adjust the step size for visualization

for bin_idx = 1:8
    plot_narrow_weights = mean_weights_Narrow_avg(bin_idx, :) + bin_idx*8;
    plot(t_vector, plot_narrow_weights, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
         'DisplayName', sprintf('Bin %d', bin_idx));
end
title('Group Average Weights (Narrow Condition)');
xlabel('Time (ms)');
ylabel('Mean Weight (a.u.)');
xlim([-100, 400])
% legend show;
hold off;
%%


%% 
% Plot for Wide condition
subplot(1,2,2)
hold on;
for bin_idx = 1:8
    plot_wide_weights = mean_weights_Wide_avg(bin_idx, :) + bin_idx*8;
    plot(t_vector, plot_wide_weights, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
         'DisplayName', sprintf('Bin %d', bin_idx));
end
title('Group Average Weights (Wide Condition)');
xlabel('Time (ms)');
ylabel('Mean Weight (a.u.)');
xlim([-100, 400])
% legend show;
% legend('location', 'eastoutside');
hold off;
