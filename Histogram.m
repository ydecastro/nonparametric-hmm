% Script to perform Nonparametric Hidden Markov Models Estimation 
% for k=2 states and the Histogram basis.
%
% For more information see the paper 
% "MINIMAX ADAPTIVE ESTIMATION OF NON-PARAMETRIC HIDDEN MARKOV MODELS" 
% by Y. De Castro and E. Gassiat and C. Lacour (JMLR)
%
% Date : 21/10/2015 (lastest version)
%
% Extended for case k>2 - 02/07/2015 by L. Lehericy
%
%% Seed's Randomness
clear all; 
close all;
rng('default');
rng(91405);                 % Randomness seed to get the same plots 
                            % as in the paper
                            % 91405 = Orsay ZIP code
                            
%% Setup
        % Parameters
        k=2;                % Number of hidden states
        n=5e4;              % Chain length
        Nchain=1;           % Number of chain samples
                            % Should be set to 1 in the DEPENDENT case.
                            
% ***********************************************************************
        % Transition probabilities
         p=0.2;              
         q=0.3;
        
         % True transition matrix
         Qstar=[1-p p; q 1-q];
        
         % Emission parameters (Beta laws)
         parambeta=[2 5; 4 2; 1.7 1.7];
         
% ***********************************************************************
        % Stationary law
        QA = [(Qstar - eye(k)) ones(k,1)];
        sol = zeros(k+1,1);
        sol(k+1) = 1;
        pistar = ((QA')\sol)';
        
        % True emission laws and their projection onto Sm
        nbpoints = 100;
        x=linspace(0,1,nbpoints);
        fstar=zeros(k,nbpoints);
        for j=1:k
            fstar(j,:)=betapdf(x,parambeta(j,1),parambeta(j,2));
        end
              
% ***********************************************************************
        % Hidden chain
        disp('%%%');
        disp(['Hidden chain of lenght ' num2str(n)]);
        disp('%%%');

        X=ones(Nchain,n);
        cumpi = cumsum(pistar);
        u = rand(Nchain,1);
        for j=2:k
            X(:,1) = X(:,1) + (j-1).*(u>=cumpi(j-1)).*(u<cumpi(j));
        end
        for a=1:n-1
            u=rand(Nchain,1);
            cumQ = cumsum(Qstar(X(:,a),:),2);
            for j=2:k
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
        
        
 % ***********************************************************************
 %           If you know the size m of the model go to Part IV !
 % ***********************************************************************
 
%% Part I: Selection of m (computation of the minimal contrast values)

        % Parameters                   
        min_m=2;            % Lower bound on the size of the approximation 
                            % (size of the sieve)
        max_m=50;           % Upper bound on the size of the approximation 
                            % (in the paper max_m=n but we will see that 50 is enough)
        
        % Initialization
        disp('Choice of M...');
        contrast_values=zeros(max_m-min_m+1,1);
        
        % Run through all models of size m (size of the sieve)
        for m=min_m:max_m

 % ***********************************************************************
 %            Spectral estimation of the transition matrix
 % ***********************************************************************
 
        % Step 1: Empirical estimators
        indicator=zeros(m,m,m);
        for a=1:m
            for b=1:m
                for c=1:m
                    indicator(a,b,c)=sum( (Y(:,1:(n-2))>(a-1)/m) .* (Y(:,1:(n-2))<=a/m) .*...
                        (Y(:,2:(n-1))>(b-1)/m) .* (Y(:,2:(n-1))<=b/m) .*...
                        (Y(:,3:n)>(c-1)/m) .* (Y(:,3:n)<=c/m ) );
                end
            end
        end
        Mchapeau=(m^(3/2))*(1/(n-2))*indicator;
        Pchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,2));
        Nchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,3));
        lchapeau=(m^(1/2))*(1/(n-2))*squeeze(sum(sum(indicator,3),2));
        
        % Step 2 : SVD
        [~,Dchapeau,U1chapeau]=svd(Pchapeau);
%         [U1chapeau,Dchapeau,~]=svd(Pchapeau);
        Uchapeau=U1chapeau(:,1:k);              % Keep the 2 largest sing. val.

        % Step 3 and 4
        Theta = genereUnitaire(k);
        eta=Uchapeau*Theta;

        Bchapeau=zeros(k,k,m);
        A = zeros(k,k,k,m);
        for b=1:m
            Bchapeau(:,:,b)=(Uchapeau'*Pchapeau*Uchapeau)\(Uchapeau'*squeeze(Mchapeau(:,b,:))*Uchapeau);
            for j=1:k
                A(j,:,:,b)=eta(b,j).*Bchapeau(:,:,b);
            end
        end
        Cchapeau = sum(A,4);

        % Step 5 and 6
        [Rchapeau,~]=eig(squeeze(Cchapeau(1,:,:)));

        Lchapeau=zeros(k,k);
        for b=1:k
            for j=1:k
                Lchapeau(j,:)=diag(Rchapeau\(squeeze(Cchapeau(j,:,:)))*Rchapeau);
            end
        end
        Ochapeau=Uchapeau*Theta*Lchapeau;

        % Step 7 (see below: Plots)

        % Step 8
        pichapeau=(Uchapeau'*Ochapeau)\Uchapeau'*lchapeau;
        pichapeau=projDeltaGenerale(pichapeau);         % proj onto Delta_k
        Qchapeau=(Uchapeau'*Ochapeau*diag(pichapeau))\Uchapeau'*Nchapeau*Uchapeau/(Ochapeau'*Uchapeau);
        for j=1:k
            Qchapeau(j,:)=projDeltaGenerale(Qchapeau(j,:)')';
        end
        
% ***********************************************************************
%  Solve empirical contrast value using CMAES (Covariance Matrix Adaptation Evolution Strategy) 
%  Find the parameters that minimizes the empirical risk encoded in 
%  empirical_contrast_general.m. The starting point is any estimation of 
%  the emission laws parameters (here we use Ochapeau: the spectral estimator solution). 
%  We make use of any estimation of the transiton matrix (here we use
%  Qchapeau: the spectral estimator solution) and the empirical laws of 3
%  consecutive observations given by Mchapeau
% ***********************************************************************

          % --------------------  Initialization --------------------------------  
          N = k*m;                              % number of objective variables/problem dimension
          xmean = reshape(Ochapeau,[N,1]);      % objective variables initial point
          sigma = 0.1;                          % coordinate wise standard deviation (step size)
          stopfitness = -100;                   % stop if fitness < stopfitness (minimization)
          stopeval = 1e4;                       % stop after stopeval number of function evaluations

          % Strategy parameter setting: Selection  
          lambda = 8+2*floor(3*log(N));         % population size, offspring number
          mu = lambda/2;                        % number of parents/points for recombination
          weights = log(mu+1/2)-log(1:mu)';     % muXone array for weighted recombination
          mu = floor(mu);        
          weights = weights/sum(weights);       % normalize recombination weights array
          mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

          % Strategy parameter setting: Adaptation
          cc = (4 + mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C
          cs = (mueff+2) / (N+mueff+5);         % t-const for cumulation for sigma control
          c1 = 2 / ((N+1.3)^2+mueff);           % learning rate for rank-one update of C
          cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
          damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma 
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
          counteval = 0;  %  
          while counteval < stopeval

            % Generate and evaluate lambda offspring
            for j=1:lambda,
              arx(:,j) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C) 
              arfitness(j) = empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, arx(:,j)); % objective function call
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
            C = (1-c1-cmu) * C ...                   % regard old matrix  
                 + c1 * (pc * pc' ...                % plus rank one update
                         + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
                 + cmu * artmp * diag(weights) * artmp'; % plus rank mu update 

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

            % Output 
            %more off;  % turn pagination off in Octave
            %disp(['Call ' num2str(counteval) ': empirical constrast= ' num2str(arfitness(1)) ' std= ' ... 
                  %num2str(sigma*sqrt(max(diag(C)))) ' conditioning= ' ...
                  %num2str(max(D) / min(D))]);
            % with long runs, the next line becomes time consuming
            out.dat = [out.dat; arfitness(1) sigma 1e5*D' ]; 
            out.datx = [out.datx; xmean'];
          end % while, end generation loop
            result_emp=empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, reshape(Ochapeau,[N,1]));
            contrast_values(m-min_m+1)=result_emp;
          
            disp(['For m=' num2str(m) ', gamma='  num2str(result_emp) ' after ' num2str(counteval) ' calls.']);          
          
        end
          
        disp('...done');
        
        save('histogram');
        
%% Part 2: Selection of m (slope heurisitic)
%clf;

load('histogram');

% ***********************************************************************
% USING CAPUSHE
% CAPUSHE (http://www.math.univ-toulouse.fr/~maugis/CAPUSHE.html)
% necessitates a .txt file 
%    Column 1: the model names (at most ten characters)
%    Column 2: the penalty shape value for each model
%    Column 3: the complexity value for each model
%    Column 4: the minimum contrast value for each model
% ***********************************************************************

m_vector = min_m:max_m;
penalty_shape = m_vector.*log(n)/n;

Columns = [m_vector; penalty_shape; m_vector; contrast_values'];
fileID = fopen('capushe.txt','w');
fprintf(fileID,'M=%i %1.12f %2i %1.12f\n',Columns); % Then run CAPUSHE !


% ***********************************************************************
% By hand
% ***********************************************************************
        rho_min=0;
        rho_max=20;
        rho_step=0.05;    

j=0;
clear('M_selectionne');
for rho=rho_min:rho_step:rho_max
    j=j+1;
    Penalized_contrast_values=contrast_values+(rho/n)*(min_m:max_m)'*log(n);
    [Min,Indice_Min]=min(Penalized_contrast_values);
    M_selectionne(j,1)=Indice_Min+min_m-1;
end

figure(1);
plot(rho_min:rho_step:rho_max,M_selectionne)
%title('Slope heuristic'); 
xlabel('$\rho$', 'Interpreter', 'Latex'); ylabel('$\hat{M}$', 'Interpreter', 'Latex'); ylim([min_m max_m-5]);

%intercept
p = polyfit((32:50)',-contrast_values(31:49),1);
slope=p(1)*n;

%slope = 5.3 so 2*slope=11 which selects m=23

figure(2);
y=p(2)+p(1)*(min_m:max_m);
plot(min_m:max_m, -contrast_values,min_m:max_m,y)
xlabel('$M$', 'Interpreter', 'Latex'); xlim([min_m max_m]);
%ylim([1.5 1.8]); 
l=legend('$-\gamma(\hat g_M)$', '$\textrm{Intercept\ }$', 'location', 'best');
set(l,'Interpreter','Latex');
    

%% Conclusion of the slope heuristic: m=23
    m=23;      
         
%% Part III: SPECTRAL ALGORITHM for the estimation of the Transition matrix

        disp('Spectral estimation...');
        % Step 1: Empirical estimators
        indicator=zeros(m,m,m);
        for a=1:m
            for b=1:m
                for c=1:m
                    %% CAS IID
                    %             Mchapeau(a,b,c)=mean( (Y(:,1)>(a-1)/m) .* (Y(:,1)<=a/m) .*...
                    %                            (Y(:,2)>(b-1)/m) .* (Y(:,2)<=b/m) .*...
                    %                            (Y(:,3)>(c-1)/m) .* (Y(:,3)<=c/m ) );

                    %% CAS DEP
                    indicator(a,b,c)=sum( (Y(:,1:(n-2))>(a-1)/m) .* (Y(:,1:(n-2))<=a/m) .*...
                        (Y(:,2:(n-1))>(b-1)/m) .* (Y(:,2:(n-1))<=b/m) .*...
                        (Y(:,3:n)>(c-1)/m) .* (Y(:,3:n)<=c/m ) );
					% La somme se fait sur n, pas sur s (l'indice s correspond a Nchain quand on considere Y_1, Y_2 et Y_3. Ici, s correspond a n).
					% Le premier ":" dans Y(:, 1:(n-2)) correspond a 1:1 puisqu'il n'y a qu'une seule chaine ici.
					% N = n-2 correspond au nombre d'echantillons.
                end
            end
        end
        Mchapeau=(m^(3/2))*(1/(n-2))*indicator;							% Y_1 Y_2 Y_3
        Pchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,2));				% Y_1 Y_3
        Nchapeau=(m)*(1/(n-2))*squeeze(sum(indicator,3));				% Y_1 Y_2
        lchapeau=(m^(1/2))*(1/(n-2))*squeeze(sum(sum(indicator,3),2));	% Y_1
        
        % Step 2 : SVD
        [~,Dchapeau,U1chapeau]=svd(Pchapeau);
%         [U1chapeau,Dchapeau,~]=svd(Pchapeau);
        Uchapeau=U1chapeau(:,1:k);              % Keep the 2 largest sing. val.

        % Step 3 and 4
        Theta = genereUnitaire(k);
        eta=Uchapeau*Theta;

        Bchapeau=zeros(k,k,m);
        A = zeros(k,k,k,m);
        for b=1:m
            Bchapeau(:,:,b)=(Uchapeau'*Pchapeau*Uchapeau)\(Uchapeau'*squeeze(Mchapeau(:,b,:))*Uchapeau);
            for j=1:k
                A(j,:,:,b)=eta(b,j).*Bchapeau(:,:,b);
            end
        end
        Cchapeau = sum(A,4);

        % Step 5 and 6
        [Rchapeau,~]=eig(squeeze(Cchapeau(1,:,:)));

        Lchapeau=zeros(k,k);
        for b=1:k
            for j=1:k
                Lchapeau(j,:)=diag(Rchapeau\(squeeze(Cchapeau(j,:,:)))*Rchapeau);
            end
        end
        Ochapeau=Uchapeau*Theta*Lchapeau;

        % Step 7 (see below: Plots)

        % Step 8
        pichapeau=(Uchapeau'*Ochapeau)\Uchapeau'*lchapeau;
        pichapeau=projDeltaGenerale(pichapeau);         % proj onto Delta_k
        Qchapeau=(Uchapeau'*Ochapeau*diag(pichapeau))\Uchapeau'*Nchapeau*Uchapeau/(Ochapeau'*Uchapeau);
               
        for j=1:k
            disp(['Projection de la ligne ' num2str(j)])
            Qchapeau(j,:) = projDeltaGenerale(Qchapeau(j,:)')';
        end
        
        disp('...done');
        
%% Part IV : Minimization of the contrast 
% Solved using CMA-ES (see above)

          disp('%%%');
          disp('Empirical risk estimation...');
          
          % --------------------  Initialization --------------------------------  
          clear('arx');                         % local variable
          N = k*m;                              % number of objective variables/problem dimension
          xmean = reshape(Ochapeau,[N,1]);      % objective variables initial point
          %sigma = 0.1;                         % coordinate wise standard deviation (step size)
          %stopfitness = -100;                  % stop if fitness < stopfitness (minimization)
          %stopeval = 1e4;                      % stop after stopeval number of function evaluations

          % Strategy parameter setting: Selection  
          lambda = 8+2*floor(3*log(N));         % population size, offspring number
          mu = lambda/2;                        % number of parents/points for recombination
          weights = log(mu+1/2)-log(1:mu)';     % muXone array for weighted recombination
          mu = floor(mu);        
          weights = weights/sum(weights);       % normalize recombination weights array
          mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

          % Strategy parameter setting: Adaptation
          cc = (4 + mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C
          cs = (mueff+2) / (N+mueff+5);         % t-const for cumulation for sigma control
          c1 = 2 / ((N+1.3)^2+mueff);           % learning rate for rank-one update of C
          cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
          damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma 
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
%%
          % -------------------- Generation Loop --------------------------------
          counteval = 0;  % the next 40 lines contain the 20 lines of interesting code 
          while counteval < stopeval

            % Generate and evaluate lambda offspring
            for kmuet=1:lambda,
              arx(:,kmuet) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C) 
              arfitness(kmuet) = empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, arx(:,kmuet)); % objective function call
              counteval = counteval+1;
            end
%%
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
            C = (1-c1-cmu) * C ...                   % regard old matrix  
                 + c1 * (pc * pc' ...                % plus rank one update
                         + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
                 + cmu * artmp * diag(weights) * artmp'; % plus rank mu update 

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

            % Output 
            more off;  % turn pagination off in Octave
            disp(['Call ' num2str(counteval) ': empirical constrast= ' num2str(arfitness(1)) ' std= ' ... 
                  num2str(sigma*sqrt(max(diag(C)))) ' conditioning= ' ...
                  num2str(max(D) / min(D))]);
            % with long runs, the next line becomes time consuming
            out.dat = [out.dat; arfitness(1) sigma 1e5*D' ]; 
            out.datx = [out.datx; xmean'];
          end % while, end generation loop

          disp('...done');
                   
          % ------------- Final Message and Plotting Figures --------------------
          disp(['CMA-ES final value after ' num2str(counteval) ' iterations : ' num2str(arfitness(1))]);
          disp(['Empirical contrast initial value: ' num2str(...
          empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, reshape(Ochapeau,[N,1])))]);
          bool=empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, xmean)<...
               empirical_contrast_general(pichapeau, Qchapeau, Mchapeau, arx(:, arindex(1)));
          xmin = arx(:, arindex(1))*(1-bool)+xmean*(bool);  % Return best point of last iteration.
                                                            % Notice that xmean is expected to be even
                                                            % better.
          figure(1); hold off; semilogy(abs(out.dat)); hold on;  % abs for negative fitness
          semilogy(out.dat(:,1) - min(out.dat(:,1)), 'k-');  % difference to best ever fitness, zero is not displayed
          title('fitness, sigma, sqrt(eigenvalues)'); grid on; xlabel('iteration');  
          figure(2); hold off; plot(out.datx); 
          title('Distribution Mean'); grid on; xlabel('iteration')
          
          
%% Plots
clf;
% Density of the emission laws.
        basehisto=zeros(m,100);         % Histogram basis
        for j=1:m
            basehisto(j,:)=(x>(j-1)/m).*(x<=j/m);
        end
        basehisto=sqrt(m)*basehisto;
		% basehisto(j,x) = 0 if x isn't in the j-th interval (open to the
		% left, closed to the right), else sqrt(m).   
        
        % True law of the observations
        Ostar = (1/100).*basehisto*(fstar');
        fm = Ostar'*basehisto;

         Otilde=reshape(xmin,[m,k]);
            
          % Labelling
            lab=norm(Ostar-Otilde)>norm(fliplr(Ostar)-Otilde);  %boolean 

            if lab 
                Otilde=fliplr(Otilde);
                Ochapeau=fliplr(Ochapeau);
            end
        
           % Estimation of the densities
           fchapeau = Ochapeau' * basehisto;
           ftilde = Otilde' * basehisto;
           
           % verification de l'estimation de O

%-------------------------------------------------------------------------
            permut = [1 2 3];       % A modifier à la main.
%-------------------------------------------------------------------------
            
            SpectralRisk = zeros(k,1);
            ContrastRisk1 = zeros(k,1);
            for j=1:k
                SpectralRisk(j)=norm(Ostar(:,j)-Ochapeau(:,permut(j)));
                ContrastRisk(j)=norm(Ostar(:,j)-Otilde(:,permut(j)));
            end
            
            strSpecRisk = num2str(SpectralRisk', '%d and ');
            disp(['Spectral L2-risks: ' strSpecRisk(1:end-4)]);
            strContRisk = num2str(ContrastRisk, '%d and ');
            disp(['Emp. cont. L2-risks: ' strContRisk(1:end-4)]);

            % graphs
            figure(4);
            for j=1:k
                subplot(1,k,j)
                hold on
                line1=plot(x,fstar(j,:),'b-');
                line3=stairs(x,fchapeau(permut(j),:),'rO-','MarkerSize',3);
                line4=stairs(x,ftilde(permut(j),:),'gx-','MarkerSize',3);
                hold off
                %legend('True density', 'Orthogonal projection', 'Spectral estimator', 'Empirical contrast est.', 'Location','Best')
                title(['Emission law ' num2str(j)])
            end
                        
            % Construct a Legend with the data from the sub-plots

            hL = legend([line1,line3,line4],{'True density', 'Spectral method', 'Empirical Contrast method'});

            % Programatically move the Legend

            newPosition = [0.35 0.6 0.2 0.2];

            newUnits = 'normalized';

            set(hL,'Position', newPosition,'Units', newUnits);