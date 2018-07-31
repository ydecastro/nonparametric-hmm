function y = projDeltaGenerale(x)
% Input : a COLUMN vector x.
% Output : a COLUMN vector y, its projection on Delta_k, the simplex in
% dimension k.
%
% We use successive projections according to orthogonal vectors in order to
% eliminate negative coordinates.
%
% Algorithm :
% indicator <- ones(k,1)    % indicator(i)=1 iff the coordinate hasn't been eliminated
% Id_k <- eye(k)
% y <- indicator/k + (Id_k - ones(k)/k)*x)
%     While i exists such that y(i)<0
%         indicator(i) <- 0
%         keff <- indicator' * indicator
%         e_i <- indicator
%         e_i(i) <- -keff
%         normalisation <- e_i' * e_i
%         tempVect <- indicator / keff
%         y <- tempVect + (Id_k-e_i*e_i'/normalisation)*(y-tempVect)
%     Return x

% Luc Lehericy, 25/06/2014

% % Test parameters
% x=[2;0.5;0];
% x=[-2;1];
% x=[0.5;0.7];
% x=[2;0.5];
%   --> Valid in dimension 2

temp = size(x);
k = temp(1);

indicator = ones(k,1);
Id_k = eye(k);
y = indicator/k + (Id_k - ones(k)/k)*x;

% Find non-eliminated negative coordinates
indNeg = find(y<0);

while numel(indNeg) > 0
    i = indNeg(1);
    
    % Eliminate coordinate i
    indicator(i) = 0;
    keff = indicator' * indicator;
    e_i = indicator;
    e_i(i) = -keff;
    
    % Projection
    normalisation = e_i' * e_i;
    tempVect = indicator / keff;
    y = tempVect + (Id_k-e_i*e_i'/normalisation)*(y-tempVect);
    
    % Find non-eliminated negative coordinates
    indNeg = find((y<0) .* indicator);
end
