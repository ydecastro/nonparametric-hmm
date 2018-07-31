% Script to perform a phase transition diagram for Non-Parametric Hidden 
% Two States Markov Models Estimation 
%
%
% For more information see the paper 
% "MINIMAX ADAPTIVE ESTIMATION OF NON-PARAMETRIC HIDDEN MARKOV MODELS" 
% by Y. De Castro and É. Gassiat and C. Lacour 
% in the case K=2 (two hidden states)
% Special thanks to Luc Lehericy for improving this code
%
% date : 10/11/2015
%% Seed's Randomness
clear all; 
close all;
rng('default');
rng(91405);                 % Randomness seed to get the same plots 
                            % as in the paper
                            % 91405 = Orsay ZIP code
                            
%% Setup
            % Parameters
            n=5e4;              % Chain length
            Nchain=1;           % Number of chain samples
                                % Should be set to 1 in the DEPENDENT case
                                % bounds
            min_m=5;            % Lower bound on the size of the approximation
                                % (size of the sieve)
            max_m=45;           % Upper bound on the size of the approximation
                                % (in the paper max_m=n but we will see that 50 is enough)
            iter=40;            % Number of loops
                                        
% ***********************************************************************
            % Transition probabilities
            p=0.2;
            q=0.3;
            
            % True transition matrix
            Qstar=[1-p p; q 1-q];
            
            % Emission parameters (Beta laws)
            parambeta=[2 5; 4 2];
         
% ***********************************************************************
            % Stationary law
            QA = [(Qstar - eye(2)) ones(2,1)];
            sol = zeros(2+1,1);
            sol(2+1) = 1;
            pistar = ((QA')\sol)';
        
            % True emission laws and their projection onto Sm
            % Number of points to compute the projection on Sm, since we
            % use Mmax=50 bins, one needs at least 50*10 points
            nbpoints = 1e4;                             
                               
            x=linspace(0,1,nbpoints);
            fstar=zeros(2,nbpoints);
            for j=1:2
                fstar(j,:)=betapdf(x,parambeta(j,1),parambeta(j,2));
            end
          
%% Loops 
        
            Spectral_Variance=zeros(max_m-min_m+1,iter);
            Contrast_Variance_SpectStart=zeros(max_m-min_m+1,iter); 
       
for m=min_m:max_m
        
        % True law of the observation
        basehisto=zeros(m,nbpoints);         % Histogram basis
        for j=1:m
            basehisto(j,:)=(x>(j-1)/m).*(x<=j/m);
        end
        basehisto=sqrt(m)*basehisto;
		% basehisto(j,x) = 0 if x isn't in the j-th interval (open to the
		% left, closed to the right), else sqrt(m).   
        
        % True law of the observations
        Ostar = (1/nbpoints).*basehisto*(fstar');
        fm = Ostar'*basehisto;
    
        disp(['M = ' num2str(m)]);
       
    for it=1:iter
                
% ***********************************************************************
        % Hidden chain
        
        X=ones(Nchain,n);
        cumpi = cumsum(pistar);
        u = rand(Nchain,1);
        for j=2:2
            X(:,1) = X(:,1) + (j-1).*(u>=cumpi(j-1)).*(u<cumpi(j));
        end
        for a=1:n-1
            u=rand(Nchain,1);
            cumQ = cumsum(Qstar(X(:,a),:),2);
            for j=2:2
                 X(:,a+1) = X(:,a+1) + (j-1).*(u>=cumQ(:,j-1)).*(u<cumQ(:,j));
            end
        end
                        
% ***********************************************************************
        % Observations
        param1 = zeros(Nchain,n);
        param2 = param1;
        for j=1:Nchain
            param1(j,:) = parambeta(X(j,:),1);
            param2(j,:) = parambeta(X(j,:),2);
        end
        Y = betarnd(param1, param2);    % (Nchain*n) matrix

%% Part I: SPECTRAL ALGORITHM

        % Step 1: Empirical estimators
        indicator=zeros(m,m,m);
        for a=1:m
            for b=1:m
                for c=1:m
                    %% Scenario B
                    indicator(a,b,c)=sum( (Y(:,1:(n-2))>(a-1)/m) .* (Y(:,1:(n-2))<=a/m) .*...
                        (Y(:,2:(n-1))>(b-1)/m) .* (Y(:,2:(n-1))<=b/m) .*...
                        (Y(:,3:n)>(c-1)/m) .* (Y(:,3:n)<=c/m ) );
                end
            end
        end
        Mchapeau=(m^(3/2))*(1/(n-2))*indicator;							% Y_1 Y_2 Y_3
        Pchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,2));				% Y_1 Y_3
        Nchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,3));				% Y_1 Y_2
        lchapeau=(m^(1/2))*(1/(n-2))*squeeze(sum(sum(indicator,3),2));	% Y_1
        
        % Step 2 : SVD
        [~,Dchapeau,U1chapeau]=svd(Pchapeau);
        Uchapeau=U1chapeau(:,1:2);              % Keep the 2 largest sing. val.

        % Step 3 and 4
        Theta = genereUnitaire(2);
        eta=Uchapeau*Theta;

        Bchapeau=zeros(2,2,m);
        A = zeros(2,2,2,m);
        for b=1:m
            Bchapeau(:,:,b)=(Uchapeau'*Pchapeau*Uchapeau)\(Uchapeau'*squeeze(Mchapeau(:,b,:))*Uchapeau);
            for j=1:2
                A(j,:,:,b)=eta(b,j).*Bchapeau(:,:,b);
            end
        end
        Cchapeau = sum(A,4);

        % Step 5 and 6
        [Rchapeau,~]=eig(squeeze(Cchapeau(1,:,:)));

        Lchapeau=zeros(2,2);
        for b=1:2
            for j=1:2
                Lchapeau(j,:)=diag(Rchapeau\(squeeze(Cchapeau(j,:,:)))*Rchapeau);
            end
        end
        Ochapeau=Uchapeau*Theta*Lchapeau;

        % Step 8
        pichapeau=(Uchapeau'*Ochapeau)\Uchapeau'*lchapeau;
        pichapeau=projDeltaGenerale(pichapeau);         % proj onto Delta_2
        Qchapeau=(Uchapeau'*Ochapeau*diag(pichapeau))\Uchapeau'*Nchapeau*Uchapeau/(Ochapeau'*Uchapeau);
               
        for j=1:2
            Qchapeau(j,:) = projDeltaGenerale(Qchapeau(j,:)')';
        end
        
        %projection pichapeau
        QB = [(Qchapeau - eye(2)) ones(2,1)];
        sol = zeros(2+1,1);
        sol(2+1) = 1;
        pichapeau = ((QB')\sol)';
        
%% Part II : Empirical contrast starting from spectral
% Solved using CMA-ES

      % --------------------  Initialization --------------------------------  
          N = 2*m;                              % number of objective variables/problem dimension
          xmean = reshape(Ochapeau,[N,1]);      % objective variables initial point
          sigma = 0.1;                          % coordinate wise standard deviation (step size)
          stopfitness = -100;                   % stop if fitness < stopfitness (minimization)
          stopeval = 1e4;                       % stop after stopeval number of function evaluations

          % Strategy parameter setting: Selection  
          lambda = 10;                          % population size, offspring number
          mu = lambda/2;                        % number of parents/points for recombination
          weights = log(mu+1/2)-log(1:mu)';     % muXone array for weighted recombination
          mu = floor(mu);        
          weights = weights/sum(weights);       % normalize recombination weights array
          mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

          % Strategy parameter setting: Adaptation
          cc = (4 + mueff/N) / (N+4 + 2*mueff/N);   % time constant for cumulation for C
          cs = (mueff+2) / (N+mueff+5);             % t-const for cumulation for sigma control
          c1 = 2 / ((N+1.3)^2+mueff);               % learning rate for rank-one update of C
          cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
          damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs;       % damping for sigma 
                                                                    % usually close to 1
          % Initialize dynamic (internal) strategy parameters and constants
          pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
          B = eye(N,N);                       % B defines the coordinate system
          D = ones(N,1);                      % diagonal D defines the scaling
          C = B * diag(D.^2) * B';            % covariance matrix C
          invsqrtC = B * diag(D.^-1) * B';    % C^-1/2 
          eigeneval = 0;                      % track update of B and D
          chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of 
                                              %   ||N(0,I)|| == norm(randn(N,1))
          out.dat = []; out.datx = [];  % for plotting output
            
          clear('arx');
          clear('arfitness');
          
      % -------------------- Generation Loop --------------------------------
          clear arx arfitness arindex
          counteval = 0;  
          while counteval < stopeval

            % Generate and evaluate lambda offspring
            for k=1:lambda,
              arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C) 
              arfitness(k) = empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, arx(:,k)); % objective function call
              counteval = counteval+1;
            end

            % Sort by fitness and compute weighted mean into xmean
            [arfitness, arindex] = sort(arfitness);  % minimization
            xold = xmean;
            xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value

            % Cumulation: Update evolution paths
            ps = (1-cs) * ps ... 
                  + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma; 
            hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2 + 4/(N+1);
            pc = (1-cc) * pc ...
                  + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma; 

            % Adapt covariance matrix C
            artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors
            C = (1-c1-cmu) * C ...                          % regard old matrix  
                 + c1 * (pc * pc' ...                       % plus rank one update
                         + (1-hsig) * cc*(2-cc) * C) ...    % minor correction if hsig==0
                 + cmu * artmp * diag(weights) * artmp';    % plus rank mu update 

            % Adapt step size sigma
            sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); 

            % Update B and D from C
            if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
              eigeneval = counteval;
              C = triu(C) + triu(C,1)'; % enforce symmetry
              [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
              D = sqrt(diag(D));        % D contains standard deviations now
              invsqrtC = B * diag(D.^-1) * B';
            end

            % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
            if arfitness(1) <= stopfitness || max(D) > 1e7 * min(D)
              break;
            end
            
          end % while, end generation loop

      %---------------------- Keep the best result ----------------------
      
          bool=empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, xmean)<...
               empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, arx(:, arindex(1)));
          xmin = arx(:, arindex(1))*(1-bool)+xmean*(bool);  % Return best point of last iteration.
                                                            % Notice that xmean is expected to be even
                                                            % better.
                                        
          Otilde=reshape(xmin,[m,2]);                       %solution point
            
      %---------------------- Labelling ---------------------------------

            if norm(Ostar-Otilde)>norm(fliplr(Ostar)-Otilde)
                Otilde=fliplr(Otilde); 
            end
            
                ContrastVar1=norm(Ostar(:,1)-Otilde(:,1));
                ContrastVar2=norm(Ostar(:,2)-Otilde(:,2));
                ContrastVar=max(ContrastVar1,ContrastVar2);
            
            Contrast_Variance_SpectStart(m-min_m+1,it)=ContrastVar;
            
            if norm(Ostar-Ochapeau)>norm(fliplr(Ostar)-Ochapeau)
                Ochapeau=fliplr(Ochapeau);
            end

                SpectralVar1=norm(Ostar(:,1)-Ochapeau(:,1));
                SpectralVar2=norm(Ostar(:,2)-Ochapeau(:,2));
                SpectralVar=max(SpectralVar1,SpectralVar2);
                    
                Spectral_Variance(m-min_m+1,it)=SpectralVar;
                   
    end
     %% Quantile plot
    figure(1);
    alpha = 0.25;       % Plot confidence band [alpha 1-alpha]

            X_plot          =   min_m:max_m;
            Data_contrast   =   Contrast_Variance_SpectStart;
            Data_spectral   =   Spectral_Variance;

            Mean_Contrast   =   median(Data_contrast');
            Q_Contrast      =   ...
                [1 0; 0 -1]*quantile(Data_contrast',[alpha 1-alpha])+[-Mean_Contrast; Mean_Contrast];

            Mean_Spectral   =   median(Data_spectral');
            Q_Spectral      =   ...
                [1 0; 0 -1]*quantile(Data_spectral',[alpha 1-alpha])+[-Mean_Spectral; Mean_Spectral];

     shadedErrorBar(X_plot,Mean_Contrast,Q_Contrast,'-b',1); 
     hold on
     shadedErrorBar(X_plot,Mean_Spectral,Q_Spectral,'-r',1); 
     hold off
     save('variances_1000pts');
end