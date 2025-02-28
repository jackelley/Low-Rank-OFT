clear

N = 200;

tstart = tic;
[UD_OFT,U_EX]=DOFT2_gauss(N);
telapsedD = toc(tstart);
D_err = norm(UD_OFT-U_EX)/norm(U_EX);
    

for Tf = 6
    DATA = [];    
    for NT = [50 100 200 400]*ceil(Tf/pi - 0.1)
        dt = Tf / NT;
        tstart = tic;
        [UFR_OFT] = FROFT2_gauss(N,dt,Tf);
        telapsedFR = toc(tstart);
        tstart = tic;
        [U_vAp, S_vAp, V_vAp,ranks]=LROFT_LRIAT2_gauss(N,dt,Tf);
        telapsedLR = toc(tstart);
        ULR_OFT = U_vAp*S_vAp*V_vAp';
        FR_err = norm(UFR_OFT-UD_OFT)/norm(U_EX);
        LR_err = norm(ULR_OFT-UD_OFT)/norm(U_EX);
        DATA = [DATA ;[dt telapsedFR telapsedLR FR_err LR_err D_err]];    
    end
    
    figure(1)
    set(gcf, 'Position',  [0, 0, 500, 400])
    set(gca,'Fontsize',20)
    loglog(DATA(:,1),DATA(:,4),'bo-',...
           DATA(:,1),DATA(:,5),'r--',...
           DATA(:,1),DATA(:,6),'k',...
           DATA(:,1),DATA(:,1).^2,'k:',...
           'linewidth',2)
    set(gca,'Fontsize',20)
    title('Errors vs Timestep')
    xlabel("Error")
    ylabel("Timestep Size")
    set(gca,'linew',2)
    legend('Full-rank','Low-rank','Direct','2nd order')
    hold on
    
    figure(2)
    set(gcf, 'Position',  [500, 0, 500, 400])
    set(gca,'Fontsize',20)
    loglog(DATA(:,4),DATA(:,2),'bo-',...
           DATA(:,5),DATA(:,3),'r--',...
           'linewidth',2)
    set(gca,'Fontsize',20)
    title('Time vs Error')
    xlabel("Time (s)")
    ylabel("Error")
    set(gca,'linew',2)
    legend('Full-rank','Low-rank')
    hold on
    %str = sprintf('EXPrank_%d_%d.eps',nn,m0);
    %print(str,'-depsc2')
    %str = sprintf('EXPrank_%d_%d.png',nn,m0);
    %print(str,'-dpng')
    drawnow
end