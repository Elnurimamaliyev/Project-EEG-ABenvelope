% Plot bin edges


dB_edges = 8 * (1:8); % Creates edges at 8, 16, ..., 64 dB
log_edges = 10 .^ (dB_edges / 20); % Convert dB to linear scale
log_edges = log_edges / max(log_edges); % Normalize to range [0, 1]

figure;
plot(log_edges)








%%

binEdges_dB = linspace(0,0.98,9);
binEdges_dB = linspace(8,64,8);

numBins = length(binEdges_dB);                                   % Number of dB bins

binEdges_amp = 10.^(binEdges_dB / 20);
normbinEdges_amp = normalize(binEdges_amp,2,'range');

figure;
plot(normbinEdges_amp)
plot(normbinEdges_amp)


%%
% Define bin edges in dB for two different ranges
binEdges_dB1 = linspace(0, 0.98, 8);  % Range from 0 to 0.98 in 9 bins
binEdges_dB2 = linspace(8, 64, 8);    % Range from 8 to 64 in 8 bins

% Convert dB bin edges to amplitude and normalize
binEdges_amp1 = 10.^(binEdges_dB1 / 20);
binEdges_amp2 = 10.^(binEdges_dB2 / 20);

% Normalize the amplitude bin edges to range [0, 1]
normbinEdges_amp1 = normalize(binEdges_amp1, 'range');
normbinEdges_amp2 = normalize(binEdges_amp2, 'range');

% Create a linearity indicator line from 0 to 1
linearity_indicator = linspace(0, 1, max(length(normbinEdges_amp1), length(normbinEdges_amp2)));

% Plot the normalized amplitude bin edges with the linearity indicator
figure;
hold on;
plot(normbinEdges_amp1, '-o', 'DisplayName', '0 to 0.98 dB Range (Our selection)');
plot(normbinEdges_amp2, '-x', 'DisplayName', '8 to 64 dB Range (Dreannan, 2019)');
plot(linearity_indicator, '--', 'DisplayName', 'Linearity Indicator', 'Color', [0.5, 0.5, 0.5]); % Dashed line for linearity

xlabel('Bin Index');
ylabel('Normalized Amplitude');
title('Normalized Amplitude for Different dB Bin Ranges with Linearity Indicator');
legend;
hold off;
