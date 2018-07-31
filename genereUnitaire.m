function U = genereUnitaire(k)
% Generates a kxk uniform unitary matrix.
%
% Draws an orthonormal basis of R^k by first drawing a vector X_1 uniformly
% on S^k-1, then a second vector X_2 uniformly on the intersection of S^k-1
% and Orthog(X_1), and so on.
%
% This is achieved by taking X_j = Y_j / norm(Y_j) where Y_j are defined
% as follow :
% Take Z_j ~ N(0, eye(k)) independent vectors, and set Y_j = P_j*Z_j where
% P_j is defined as the orthogonal projection on Orthog(X_1, ..., X_j-1).
% Note that P_j : P_j = P_j-1 * (eye(k) - X_j*X_j') where P_0 = eye(k).
%
% (X_j) defines an orthonormal basis of X_k. Set U the matrix whose j-th
% column is X_j. Then U is a uniform unitary matrix.

% Luc Lehericy, 25/06/2014

P = eye(k);
U = zeros(k,k);

for j=1:k
    Y = P * randn(k,1);
    X = Y / norm(Y);
    U(:,j) = X;
    P = P * ( eye(k) - X*X' );
end