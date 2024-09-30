function [stim_ABenv_Norm, stim_ABenv, NormEnv, NormBinEdges, BinEdges] = ABenvelopeGenerator_V2(stim_Env, binEdges_dB, num_bins)

    % Amplitude binned envelope
    % functionAmplitude_binned_vol1
    
    % Feature extraction and amplitude binning
    
    % Define the number of bins for amplitude binning
    if nargin < 2
        num_bins = 8; % Number of bins, adjust as needed
    end 


    %% Logarifmic
    % Compute the amplitude range for binning
    % min_amp = min(stim_Env(:));
    % max_amp = max(stim_Env(:));
    % bin_edges = linspace(min_amp, max_amp, num_bins + 1)
    % BinEdges = logspace(log10(min_amp), log10(max_amp), num_bins + 1);
    
    %% Linear 
    
    % Transfer it to linear scale 
    binEdges_linear = 10.^(binEdges_dB / 20);
    BinEdges= binEdges_linear';

    % Normalize
    NormEnv = normalize(stim_Env,1,'range');
    NormBinEdges = normalize(BinEdges,1,'range');

    % Initialize the binned envelope matrix
    stim_ABenv = zeros(size(NormEnv,1), num_bins);
    
    % Bin the envelope data
    for bin_idx = 1:num_bins
        BinMask = (NormEnv >= NormBinEdges(bin_idx)) & (NormEnv < NormBinEdges(bin_idx + 1));

        % sum non-zero nums for differrent bins 
        stim_ABenv(BinMask, bin_idx) = NormEnv(BinMask);
    end
    
    stim_ABenv_Norm = normalize(stim_ABenv,1,'range');
    
end




