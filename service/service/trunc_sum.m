function [U, S, V] =trunc_sum(C, tol, rmax)
% return the truncated sum of low rank matrix stored in C
% C has three cols, Ui, Si, Vi
% tol is tolerance in Frob norm
% rmax is max rank

nmat=size(C,1);
r = zeros(nmat,1);
% this is nx, ny
n1 = size(C{1,1},1);
n2 = size(C{1,3},1);
for i =1:nmat
    r(i) = size(C{i,1},2);
end
% sum of total rank
rtot = sum(r);

bigU=zeros(n1,rtot);
bigV=zeros(n2,rtot);
bigS=zeros(rtot,rtot);

rc = [0; cumsum(r)];
for i=1:nmat
    bigU(:,rc(i)+1:rc(i+1)) = C{i,1};
    bigV(:,rc(i)+1:rc(i+1)) = C{i,3};
    bigS(rc(i)+1:rc(i+1),rc(i)+1:rc(i+1))=C{i,2};
end

[QU,RU,PU] = qr(bigU,'econ');
[QV,RV,PV] = qr(bigV,'econ');
[Ust,Sst,Vst] = svd(RU*PU'*bigS*PV*RV','econ');
sd = diag(Sst);
energy = cumsum(sd(end:-1:1).^2);
r_st = length(energy) - length(find(energy < tol^2));
r_st = min(r_st, rmax);
U=QU*Ust(:,1:r_st);
V=QV*Vst(:,1:r_st);
S=Sst(1:r_st,1:r_st);
