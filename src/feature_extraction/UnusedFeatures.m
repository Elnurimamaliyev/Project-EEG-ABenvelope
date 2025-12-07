% Combine other features (e.g., )
%%% Unused
CmbOnsABenvConc = [stim_ABenv(1:2,:) + stim_Onset; stim_ABenv(2:end,:)];  % ABEnvelope and Onset Concatinated (BinNum, all)
Com_Env_Onset_Concd = [stim_Env; stim_Onset]; % Env_Onset - Envelope and Onset Concatinated (2, all)
Com_Env_Onset_plus = stim_Env+ stim_Onset; % Env_Onset - Envelope and Onset Concatinated (2, all)
Com_OnsetPlusABenv = stim_ABenv + stim_Onset;
Ons_Env = [stim_Onset/30;stim_Env]; % Onset Envelope (Onset + Envelope) - Dividing to onset envelope to fractions to just emphasize onsets in Envelope
Normalized_Onset20x_ABenvelope = normalize(Onsetx23_plus_ABenvelope,2,'range'); % The Best so far
Norm_Env_Onset = Env_Onset ./ max(Env_Onset, [], 2); 
Onset23x_conc_ABenvelope = [stim_ABenv; stim_Onset];