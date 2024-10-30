function [norm_onsetEnvelope, ABenvNorm, Onset, onsetEnvelope, resp] = feature_deriv(audio_dat, fs_audio, fs_eeg, resp)

%% Compute the analytic signal using the Hilbert transform
envelopemodel = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox

% Size check (if required, adapt this function as needed)
[envelopemodel, resp] = size_check(envelopemodel, resp);  % Ensure matching sizes

EnvNorm = (envelopemodel - min(envelopemodel)) / (max(envelopemodel) - min(envelopemodel));


onset_threshold = 0.5; % Default threshold for peak detection
min_peak_distance = 90; % Default minimum distance between peaks
    
Onset= OnsetGenerator(EnvNorm, min_peak_distance, onset_threshold);
% Onset Envelope
onsetEnvelopeDiff = [0; diff(EnvNorm)];  % Calculate difference and prepend 0 for consistent length
onsetEnvelope = max(onsetEnvelopeDiff, 0);  % Keep only positive values, set negatives to 0

norm_onsetEnvelope = normalize(onsetEnvelope,1,'range');

% %% Parameters
% step_size = 4;
% binEdges_dB = 0:step_size:28;                          % Define bin edges in dB (8, 16, 24, ..., 64)
% step_size=0.98/8;

binEdges_dB = linspace(0,0.98,9);
numBins = length(binEdges_dB);                                   % Number of dB bins
% Convert envelope to decibel (dB) scale
% envelope_dB = 20 * log10(envelopemodel + eps);  % Add eps to avoid log(0)

% envelope_dB = real(envelope_dB); % Uncomment if you want to ensure real values

% Calculate bin edges in amplitude scale (logarithmic binning)
binEdges_amp = 10.^(binEdges_dB / 20);
% binEdges_ampNorm = normalize(binEdges_amp, 2,'range');

% Initialize matrix for binned envelopes
binnedEnvelopes_dB = zeros(size(EnvNorm, 1), numBins);

% Normalize the envelope in dB
% min_value = min(envelope_dB);                          % Minimum value of envelope_dB
% shifted_envelope_dB = envelope_dB - min_value;        % Shift to make positive

% normShifted_envelope_dB = normalize(shifted_envelope_dB,1,'range');
normbinEdges_amp = normalize(binEdges_amp,2,'range');

% Calculate the histogram counts and bin indices (using histcounts)
[~, ~, binMask] = histcounts(EnvNorm, normbinEdges_amp);

% Binned envelope: Populate binnedEnvelopes_dB based on the bin indices
for i = 1:length(binMask)
    if binMask(i) > 0  % Check to ensure binMask is valid (not zero)
        binnedEnvelopes_dB(i, binMask(i)) = EnvNorm(i);
    end
end

% Normalize the binned envelopes between 0 and 1
ABenvNorm = normalize(binnedEnvelopes_dB, 1, "range");


% time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
% time_env = (0:length(envelopemodel) - 1) / fs_eeg;   % Time array for envelope and binned envelope
% window_time = [945 955];

% Colors
% blue = [0 0.4470 0.7410];
% orange = [0.8500 0.3250 0.0980];
% yellow = [0.9290 0.6940 0.1250];

% figure;
% 
% % Subplot 1: Original Audio Signal
% subplot(5, 1, 1);
% plot(time_audio, audio_dat, 'color','k', 'DisplayName', 'Original Audio');  % Original audio
% title('Original Audio Signal');
% xlabel('Time (s)');
% xlim(window_time);
% grid on;
% 
% % Subplot 2: Standard Envelope
% subplot(5, 1, 2);
% plot(time_env, EnvNorm, 'color', blue, 'DisplayName', 'Standard Envelope');  % Standard envelope
% title('Standard Envelope');
% xlabel('Time (s)');
% xlim(window_time);
% grid on;
% 
% % Subplot 3: SPL Envelope
% subplot(5, 1, 3);
% plot(time_env, normSPLenvelope, 'color',orange, 'DisplayName', 'SPL Envelope');  % SPL envelope
% title('SPL Envelope');
% xlabel('Time (s)');
% ylabel('SPL (dB)');
% xlim(window_time);
% grid on;
% 
% % Subplot 4: Onset Envelope
% subplot(5, 1, 4);
% plot(time_env, onsetEnvelope, 'color', yellow, 'DisplayName', 'Onset Envelope');  % Onset envelope
% title('Onset Envelope');
% xlabel('Time (s)');
% xlim(window_time);
% grid on;
% 
% % Subplot 5: Combined Plot
% subplot(5, 1, 5);
% hold on;  % Hold for multiple plots
% plot(time_audio, audio_dat, 'color','k', 'DisplayName', 'Original Audio');  % Original audio
% plot(time_env, EnvNorm, 'color', blue, 'DisplayName', 'Standard Envelope');  % Standard envelope
% plot(time_env, normSPLenvelope, 'color',orange, 'DisplayName', 'SPL Envelope');  % SPL envelope
% plot(time_env, onsetEnvelope, 'color',yellow, 'DisplayName', 'Onset Envelope');  % Onset envelope
% 
% title('Combined Plot of Audio and Envelopes');
% xlabel('Time (s)');
% ylabel('Amplitude');
% xlim(window_time);
% legend('Location', 'best');  % Add legend
% grid on;  % Add grid for better visibility
% hold off;  % Release the hold
% 
% % Adjust figure layout for better spacing
% sgtitle('Audio Signals and Envelopes');  % Overall title for the figure


% Plot Binned Envelopes
% figure;
% % Subplot 5: Combined Plot
% subplot(2, 1, 1);
% hold on;  % Hold for multiple plots
% % plot(time_audio, audio_dat, 'color','k', 'DisplayName', 'Original Audio');  % Original audio
% plot(time_env, EnvNorm, 'color', blue, 'DisplayName', 'Standard Envelope');  % Standard envelope
% plot(time_env, Onset, 'color',orange, 'DisplayName', 'Onset');  % SPL envelope
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

% norm_audio_dat = normalize(audio_dat,2,"range",[-1 1]);
% %%
% window_time = [1149 1151];
% 
% % Plot Binned Envelopes
% figure;
% subplot(2,1,1)
% hold on;  % Hold for multiple plots
% 
% plot(time_audio, norm_audio_dat, 'color', 'k', 'LineWidth', 1, 'DisplayName', 'Original Audio (normalized)');  % Original audio with thicker line
% plot(time_env, EnvNorm, 'color', blue, 'LineWidth', 2, 'DisplayName', 'mTRFenvelope (normalized)');  % Standard envelope with thicker line
% 
% title('Audio and Envelope');
% xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
% legend('Location', 'best');  % Add legend
% grid on;  % Add grid for better visibility
% hold off;  % Release the hold
% set(gca, 'FontSize', 13);
% box off
% 
% subplot(2,1,2)
% plot(time_env, EnvNorm, 'color', blue, 'LineWidth', 2, 'DisplayName', 'mTRFenvelope (normalized)');  % Standard envelope with thicker line
% xlim(window_time);
% legend('Location', 'best');  % Add legend
% xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
% grid on;  % Add grid for better visibility
% set(gca, 'FontSize', 13);
% title('Output Envelope');
% box off
% %% 
% figure;
% subplot(2,1,1)
% plot(time_env, EnvNorm, 'color', blue, 'LineWidth', 2, 'DisplayName', 'mTRFenvelope (normalized)');  % Standard envelope with thicker line
% xlim(window_time);
% legend('Location', 'best');  % Add legend
% xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
% grid on;  % Add grid for better visibility
% set(gca, 'FontSize', 13);
% title('Output Envelope');
% box off
% % %%
% multiplystep_size = step_size*0.9;
% subplot(2,1,2)
% for i = 1:numBins
%     plot_norm_binnedEnvelopes_dB = (ABenvNorm(:, i)*multiplystep_size) + binEdges_dB(i);
%     hold on
%     plot(time_env, plot_norm_binnedEnvelopes_dB, 'LineWidth', 2);
% end
% title(['Binned Envelope (Bin ' num2str(i) ')']);
% xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
% grid on;  % Add grid for better visibility
% box off
% xlim(window_time);
% set(gca, 'FontSize', 13);


%%
% % Parameters
% hannWindowLength = 2048;      % Hann window length
% hannOverlap = 128;            % Overlap between windows
% gamma = 10;                   % Gamma scaling factor for compression
% threshold = 0.5;              % Threshold for peak detection (adjust as needed)
% 
% %%
% % Step 1: Square the audio waveform to emphasize energy
% energy = audio_dat.^2;
% %%
% % Step 2: Smooth using Hann window
% hannWindow = hann(hannWindowLength);
% smoothedEnergy = buffer(energy, hannWindowLength, hannOverlap, 'nodelay');
% smoothedEnergy = sum(smoothedEnergy .* hannWindow);
% %%
% % Step 3: Logarithmic compression
% compressedEnergy = log(1 + gamma * smoothedEnergy);
% %%
% % Step 4: Compute the difference (first-order derivative approximation)
% diffEnergy = diff(compressedEnergy);
% smoothedDiffEnergy = filter(hann(hannWindowLength), 1, diffEnergy);
% 
% % Step 5: Half-wave rectification to retain only positive changes
% halfRectified = max(smoothedDiffEnergy, 0);
% 
% % Step 6: Detect peaks that exceed threshold to mark onsets
% onsets = halfRectified > threshold;
end