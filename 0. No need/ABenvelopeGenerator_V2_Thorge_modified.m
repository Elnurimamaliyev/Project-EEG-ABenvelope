function [env_bin_norm, env_bin, binEdges_normalized, binEdges_dB] = ABenvelopeGenerator_V2_Thorge_modified(menv_norm, binEdges_dB)
        % Define the bin edges in dB
        
            % Define the number of bins for amplitude binning
        if nargin < 2
            binEdges_dB = 8:9:64;  % Binning in 9 dB steps up to 64 dB (adjust as needed)
            % binEdges_dB = 30:8:80;  % Binning in 8 dB steps up to 64 dB (adjust as needed)
        end 

        
        % Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
        binEdges_linear = 10.^(binEdges_dB / 20);
        
        % Normalize the bin edges to be between 0 and 1
        binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
                              (max(binEdges_linear) - min(binEdges_linear));
        
        % Calculate the histogram counts and bin indices using histcounts
        [~, ~, binIndices] = histcounts(menv_norm, binEdges_normalized);
        
        % binned envelope
        env_bin = zeros(length(binEdges_dB),length(menv));
        
        for i = 1:length(binIndices)
            env_bin(binIndices(i),i) = menv_norm(i);
        end
        
        %normalize each bin 
        env_bin_norm_non_transposed = normalize(env_bin,2,'range');
        
        env_bin_norm = env_bin_norm_non_transposed';

end