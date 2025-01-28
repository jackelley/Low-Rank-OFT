function [U, S, V] = truncsum2(C, tol, rmax)
% This function returns the truncated sum of low rank matrix 
% stored in the cell C
% C has three parts, Ui, Si, Vi
% tol is tolerance in Frob norm
% rmax is max rank
% The rank of the returned matrix is always at least one 
% and as a consequence the returned matrix may only 
% contain a single singular value smaller than tol
% This is to make sure that the zero function can be represented

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
r_st = max(min(r_st, rmax),1); % Make sure the rank is at least 1
U=QU*Ust(:,1:r_st);
V=QV*Vst(:,1:r_st);
S=Sst(1:r_st,1:r_st);