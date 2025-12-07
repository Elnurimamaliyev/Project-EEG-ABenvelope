function [stim_Env, resp]= size_check(stim_Env, resp)

    % Ensure stim_Env and resp have the same number of samples by cutting the larger one
    if size(stim_Env, 1) > size(resp, 1)
        % Cut stim_Env to match the size of resp
        stim_Env = stim_Env(1:size(resp, 1), :);
    elseif size(resp, 1) > size(stim_Env, 1)
        % Cut resp to match the size of stim_Env
        resp = resp(1:size(stim_Env, 1), :);
    end

    if ~isequal(size(stim_Env,1),size(resp,1))
        error(['STIM and RESP arguments must have the same number of '...
            'observations.']);
    end


end
