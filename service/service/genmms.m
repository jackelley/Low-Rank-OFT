clear

syms t a x y k
f = exp(sin(a*sin(t)));

%matlabFunction(diff(f,t,0),"File","myfile0");
%matlabFunction(diff(f,t,1),"File","myfile1");
%matlabFunction(diff(f,t,2),"File","myfile2");
%matlabFunction(diff(f,t,3),"File","myfile3");
%matlabFunction(diff(f,t,4),"File","myfile4");
%matlabFunction(diff(f,t,5),"File","myfile5");
%matlabFunction(diff(f,t,6),"File","myfile6");
%matlabFunction(diff(f,t,7),"File","myfile7");

%f2 = exp(sin(k*x*y*sin(t)));
%matlabFunction(diff(f2,x,x,t),"File","uxxt");
%matlabFunction(diff(f2,y,y,t),"File","uyyt");

f3 = exp(sin(x+t))*exp(cos(y-t))+t*x*y;
matlabFunction(f3,"File","u");
matlabFunction(diff(f3,t),"File","ut");
matlabFunction(diff(f3,t,t),"File","utt");
matlabFunction(diff(f3,x,x),"File","uxx");
matlabFunction(diff(f3,y,y),"File","uyy");
matlabFunction(diff(f3,x,x,t),"File","uxxt");
matlabFunction(diff(f3,y,y,t),"File","uyyt");

