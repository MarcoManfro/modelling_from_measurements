function [Phi, Lambda, b] = DMD (X, X1, r)

[U,S,V] = svd(X, 'econ');
U_r = U(:,1:r);
S_r = S(1:r,1:r);
V_r = V(:,1:r);
Atilde = U_r'*X1*V_r/S_r;
[W,Lambda] = eig(Atilde);
Phi = X1*V_r/S_r*W;

alpha = S_r*(V_r(1,:)).';

b = W*Lambda\alpha;