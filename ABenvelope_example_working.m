%% Modification for numbers in beetween
amp_conti = [2, 3.2, 4.1, 0.1, 3.3, 4.4, 5.5, 44.5, 0.6, 8, 0, 3, 0, 0.6, 4.3, 5, 1, 3, 9, 5, 7, ...
    6, 1, 8, 8.5, 11, 2, 13.5, 2, 7, 12, 12, 12, 11, 10];

bins = 0:8:64;
%%
load('short_audio.mat')
load('short_envelope.mat')

audio = short_audio;
amp_conti = short_envelope';
bins = [0, 8, 16, 24, 32, 40, 48, 56, 64];
%%

[plot_matrix_bin, matrix_bin] = ABenvelope(amp_conti, bins);

% identify max and min value
min_val = min(amp_conti); max_val = max(bins);

% time array
t = 1:length(amp_conti);

% Plot Modified Amplitude Levels
% figure;
subplot(3, 1, 1);
plot(t, audio);
title('Amplitude over Time');
xlabel('Time');
ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);

subplot(3, 1, 2);
plot(t, amp_conti);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);

subplot(3, 1, 3);
plot(t, plot_matrix_bin);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([min_val-1, max_val+4]);



%% function
function [plot_matrix_bin, matrix_bin] = ABenvelope(amp_conti, bins)
    % Initialize variables
    matrix_bin = zeros(length(bins), length(amp_conti));
    % Update the matrix_bin based on the given logic
    for step_counter = 1:length(amp_conti)
        amp_value = round(amp_conti(step_counter));         % Round amplitude value to nearest integer
        [~, closest_bin_idx] = min(abs(bins - amp_value));  % Find the closest bin to the amplitude value
        matrix_bin(closest_bin_idx, step_counter) = matrix_bin(closest_bin_idx, step_counter) + 1; % Add 1 to the current amplitude value
    end

    % Populate the matrix_bin with amplitude values
    plot_matrix_bin = repmat(bins(:), 1, length(amp_conti)); 
    plot_matrix_bin = plot_matrix_bin + matrix_bin;
end

%% Working Code
% amp_conti = [2, 3.2, 4.1, 0.1, 3.3, 4.4, 5.5, 44.5, 0.6, 8, 0, 3, 0, 0.6, 4.3, 5, 1, 3, 9, 5, 7, ...
%     6, 1, 8, 8.5, 11, 2, 13.5, 2, 7, 12, 12, 12, 11, 10];
% 
% bins = 0:8:64;
% 
% 
% % Initialize variables
% matrix_bin = zeros(length(bins), length(amp_conti)); % 35 rows for 13 amplitudes, columns for time points
% 
% % Update the matrix_bin based on the given logic
% for step_counter = 1:length(amp_conti)
%     amp_value = round(amp_conti(step_counter)); % Round amplitude value to nearest integer
% 
%     % Find the closest bin to the amplitude value
%     [~, closest_bin_idx] = min(abs(bins - amp_value));
%     % closest_bin_value = bins(closest_bin_idx);
% 
%     if amp_value < 2
%         matrix_bin(1, step_counter) = matrix_bin(1, step_counter) + 1; % Add 1 to the current amplitude value
% 
%     elseif amp_value <= 64 && amp_value >= 2.0 % Ensure amplitude value is within the valid range
% 
%             matrix_bin(closest_bin_idx, step_counter) = matrix_bin(closest_bin_idx, step_counter) + 1; % Add 1 to the current amplitude value
%     end
% end
% 
% 
% % Populate the matrix_bin with amplitude values
% plot_matrix_bin = repmat(bins(:), 1, length(amp_conti));
% plot_matrix_bin = plot_matrix_bin + matrix_bin;
% 
% % identify max and min value
% min_val = min(amp_conti);
% max_val = max(bins);
% 
% % time array
% t = 1:length(amp_conti);
% 
% % Plot Modified Amplitude Levels
% figure;
% subplot(3, 1, 1);
% plot(t, amp_conti);
% title('Amplitude over Time');
% xlabel('Time');
% ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);
% 
% subplot(3, 1, 2);
% plot(t, matrix_bin);
% title('Modified Amplitude Levels over Time');
% xlabel('Time');
% ylabel('Amplitude');
% % ylim([min_val-1, max_val+4]);
% 
% subplot(3, 1, 3);
% plot(t, plot_matrix_bin);
% title('Modified Amplitude Levels over Time');
% xlabel('Time');
% ylabel('Amplitude');
% ylim([min_val-1, max_val+4]);

%% cleannes
clear; clc;
