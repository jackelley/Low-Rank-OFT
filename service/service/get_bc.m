function [BC_U,BC_V] = get_bc(nx,ny,gbottom,gtop,gleft,gright,...
                                 BStx,IPx,Ex,sigx,BSty,IPy,Ey,sigy,x,y,t)


SAT_U = IPx*(BStx*Ex-sigx*Ex);
SAT_V = IPy*(BSty*Ey-sigy*Ey);
nu = nnz(BStx);
nv = nnz(BSty);

BC_U = zeros(nx,nu+nv);
BC_V = zeros(ny,nu+nv);
for i = 1:nu/2
    BC_U(i,i) = 1;
end
for j = 1:ny
    BC_V(j,1:nu/2) = -SAT_U(1:nu/2,1)'*gleft(y(j),t);
end
for i = 1:nu/2
    BC_U(nx-i+1,nu-i+1) = 1;
end
for j = 1:ny
    BC_V(j,nu:-1:nu/2+1) = -SAT_U(nx:-1:nx-nu/2+1,nx)'*gright(y(j),t);
end

for i = 1:nv/2
    BC_V(i,nu+i) = 1;
end
for i = 1:nx
    BC_U(i,nu+[1:nv/2]) = -SAT_V(1:nv/2,1)'*gbottom(x(i),t);
end
for i = 1:nv/2
    BC_V(ny-i+1,nu+nv-i+1) = 1;
end
for i = 1:nx
    BC_U(i,nu+[nv:-1:nv/2+1]) = -SAT_V(ny:-1:ny-nv/2+1,ny)'*gtop(x(i),t);
end
end
