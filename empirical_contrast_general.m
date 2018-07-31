% ------Empirical Contrast Function---------------  
function gamma=empirical_contrast_general(pi,Q,M,A)
%pi: k*1 stationary distribution
%Q:  k*k transition matrix
%M:  m*m*m empirical joint distribution
%A:  2m*1 coeff of emission laws A=[coeff f1, coeff f2]'
maux=size(M,1);
kaux=size(Q,1);
Oaux=reshape(A,[maux,kaux]);
gamma=0;

for a=1:maux
    for b=1:maux
        for c=1:maux
            sommeEmpirique = 0;
            for k1=1:kaux
                for k2=1:kaux
                    for k3=1:kaux
                        sommeEmpirique = sommeEmpirique + pi(k1)*Q(k1,k2)*Q(k2,k3)*Oaux(a,k1)*Oaux(b,k2)*Oaux(c,k3);
                    end
                end
            end
            gamma = gamma + sommeEmpirique*(sommeEmpirique-2*M(a,b,c));
        end
    end
end