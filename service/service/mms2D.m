function out = mms2D(x,y,t,idx,idy,idt,ifun)

    if ifun == 1
        kx = 0.38*pi;
        ky = 0.4*pi;
        % T
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = sin(kx*x)*sin(ky*y)*exp(t);
            % T_xx
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = -kx^2*sin(kx*x)*sin(ky*y)*exp(t);
            % T_yy
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = -ky^2*sin(kx*x)*sin(ky*y)*exp(t);
            % T_t
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = sin(kx*x)*sin(ky*y)*exp(t);
            % T_tt
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = sin(kx*x)*sin(ky*y)*exp(t);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 2
        kx = 1.1*pi;
        ky = 1.1*pi;
        kt = 3;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = sin(kx*x+ky*y+kt*t);
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = -kx^2*sin(kx*x+ky*y+kt*t);
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = -ky^2*sin(kx*x+ky*y+kt*t);
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = kt*cos(kx*x+ky*y+kt*t);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = -kt^2*sin(kx*x+ky*y+kt*t);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 3
        kt = 1;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = x*y*exp(t);
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = 0;
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = 0;
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = x*y*exp(t);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = x*y*exp(t);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 4
        k = 1;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = exp(cos(k*x*y))*exp(t);
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = k^2*y^2*(sin(k*x*y)^2-cos(k*x*y))*exp(cos(k*x*y))*exp(t);
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = k^2*x^2*(sin(k*x*y)^2-cos(k*x*y))*exp(cos(k*x*y))*exp(t);
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = exp(cos(k*x*y))*exp(t);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = exp(cos(k*x*y))*exp(t);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 5
        k = 5;
        a = x*y*k;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = exp(cos(k*x*y*t));
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = k^2*t^2*y^2*(sin(k*t*x*y)^2-cos(k*t*x*y))*exp(cos(k*t*x*y));
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = k^2*t^2*x^2*(sin(k*t*x*y)^2-cos(k*t*x*y))*exp(cos(k*t*x*y));
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = -k*x*y*sin(k*t*x*y)*exp(cos(k*t*x*y));
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = k^2*y^2*x^2*(sin(k*t*x*y)^2-cos(k*t*x*y))*exp(cos(k*t*x*y));
        elseif (idx == 0) && (idy == 0) && (idt == 3)
            out = 1/2*a^3*sin(2*a*t)*exp(cos(a*t))*(cos(a*t) + 3);
        elseif (idx == 0) && (idy == 0) && (idt == 4)
            out = 1/8*a^4*exp(cos(a*t))*(-4*cos(a*t) + 24*cos(2*a*t) + 12*cos(3*a*t) + cos(4*a*t) - 1);
        elseif (idx == 0) && (idy == 0) && (idt == 5)
            out = -1/8*a^5*sin(a*t)*exp(cos(a*t))*(100*cos(a*t) + 96*cos(2*a*t) + 20*cos(3*a*t) + cos(4*a*t) + 31);
        elseif (idx == 0) && (idy == 0) && (idt == 6)
            out = -1/32*a^6*exp(cos(a*t))*(-148*cos(a*t) + 191*cos(2*a*t) + 630*cos(3*a*t) + 254*cos(4*a*t) + 30*cos(5*a*t) + cos(6*a*t) + 34);
        elseif (idx == 0) && (idy == 0) && (idt == 7)
            out = 1/16*a^7*sin((a*t)/2)*exp(cos(a*t))*cos((a*t)/2)*(2660*cos(a*t) + 4271*cos(2*a*t) + 2674*cos(3*a*t) + 554*cos(4*a*t) + 42*cos(5*a*t) + cos(6*a*t) + 1926);
        elseif (idx == 2) && (idy == 0) && (idt == 1)
            out = k^2*t*y^2*sin(k*t*x*y)*(k*t*x*y-k*t*x*y*sin(k*t*x*y)^2+2*sin(k*t*x*y))*exp(cos(k*t*x*y))...
                  +k^2*t*y^2*(3*k*t*x*y*sin(k*t*x*y)-2)*exp(cos(k*t*x*y))*cos(k*t*x*y);
        elseif (idx == 0) && (idy == 2) && (idt == 1)
            out = k^2*t*x^2*sin(k*t*x*y)*(k*t*x*y-k*t*x*y*sin(k*t*x*y)^2+2*sin(k*t*x*y))*exp(cos(k*t*x*y))...
                  +k^2*t*x^2*(3*k*t*x*y*sin(k*t*x*y)-2)*exp(cos(k*t*x*y))*cos(k*t*x*y);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 6
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = exp(sin(x*t))*exp(cos(y*t))+t*x*y;
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = (-t^2*sin(x*t)+(t*cos(x*t))^2)*exp(sin(x*t))*exp(cos(y*t));
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = (-t^2*cos(y*t)+(t*sin(y*t))^2)*exp(sin(x*t))*exp(cos(y*t));
        elseif (idx == 2) && (idy == 0) && (idt == 1)
            t2 = t.*x;
            t3 = t.*y;
            t5 = cos(t2);
            t6 = cos(t3);
            t7 = sin(t2);
            t8 = sin(t3);
            t9 = exp(t6);
            t10 = exp(t7);
            t11 = t5.^2;
            out = t.*t7.*t9.*t10.*-2.0+t.*t9.*t10.*t11.*2.0+t.*t2.*t5.^3.*t9.*t10-t.*t2.*t5.*t9.*t10-t.*t2.*t5.*t7.*t9.*t10.*3.0+t.*t3.*t7.*t8.*t9.*t10-t.*t3.*t8.*t9.*t10.*t11;
        elseif (idx == 0) && (idy == 2) && (idt == 1)
            t2 = t.*x;
            t3 = t.*y;
            t5 = cos(t2);
            t6 = cos(t3);
            t7 = sin(t2);
            t8 = sin(t3);
            t9 = exp(t6);
            t10 = exp(t7);
            t11 = t8.^2;
            out = t.*t6.*t9.*t10.*-2.0+t.*t9.*t10.*t11.*2.0-t.*t3.*t8.^3.*t9.*t10+t.*t3.*t8.*t9.*t10-t.*t2.*t5.*t6.*t9.*t10+t.*t3.*t6.*t8.*t9.*t10.*3.0+t.*t2.*t5.*t9.*t10.*t11;
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = (x*cos(x*t)-y*sin(y*t))*exp(sin(x*t))*exp(cos(y*t))+x*y;
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            t2 = t.*x;
            t3 = t.*y;
            t4 = x.^2;
            t5 = y.^2;
            t6 = cos(t2);
            t7 = cos(t3);
            t8 = sin(t2);
            t9 = sin(t3);
            t10 = exp(t7);
            t11 = exp(t8);
            out = -t4.*t8.*t10.*t11-t5.*t7.*t10.*t11+t4.*t6.^2.*t10.*t11+t5.*t9.^2.*t10.*t11-t6.*t9.*t10.*t11.*x.*y.*2.0;
        elseif (idx == -1) && (idy == -1) && (idt == 0)
            display('Derivative not implemented')
            return
        elseif (idx == -1) && (idy == -1) && (idt == 1)
            display('Derivative not implemented')
            return
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 7
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = x*y*exp(-t);
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            out = 0;
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            out = 0;
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            out = -x*y*exp(-t);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            out = x*y*exp(-t);
        else
            display('Derivative not implemented')
            return
        end
    elseif ifun == 8
        k = 2;
        a = x*y*k;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out = exp(-sin(a.*t.*(t-1.0).*4.0));
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            t2 = t-1.0;
            t3 = a.*t.*t2.*4.0;
            out = -exp(-sin(t3)).*cos(t3).*(a.*t.*4.0+a.*t2.*4.0);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            t2 = a.*t.*4.0;
            t3 = t-1.0;
            t4 = a.*t3.*4.0;
            t5 = t2.*t3;
            t6 = cos(t5);
            t7 = sin(t5);
            t8 = t2+t4;
            t9 = -t7;
            t11 = t8.^2;
            t10 = exp(t9);
            out = t6.^2.*t10.*t11-a.*t6.*t10.*8.0+t7.*t10.*t11;
        elseif (idx == 0) && (idy == 0) && (idt == 3)

            t2 = a.*t.*4.0;
            t3 = t-1.0;
            t4 = a.*t3.*4.0;
            t5 = t2.*t3;
            t6 = cos(t5);
            t7 = sin(t5);
            t8 = t2+t4;
            t9 = -t7;
            t11 = t8.^3;
            t10 = exp(t9);
            out = -t6.^3.*t10.*t11+t6.*t10.*t11+a.*t7.*t8.*t10.*2.4e+1-t6.*t7.*t10.*t11.*3.0+a.*t6.^2.*t8.*t10.*2.4e+1;
        elseif (idx == 0) && (idy == 0) && (idt == 4)
            t2 = a.^2;
            t3 = a.*t.*4.0;
            t4 = t-1.0;
            t5 = a.*t4.*4.0;
            t6 = t3.*t4;
            t7 = cos(t6);
            t8 = sin(t6);
            t10 = t3+t5;
            t9 = t7.^2;
            t11 = -t8;
            t13 = t10.^2;
            t12 = exp(t11);
            t14 = t13.^2;
            out = t8.^2.*t12.*t14.*3.0+t9.^2.*t12.*t14+t2.*t8.*t12.*1.92e+2+t2.*t9.*t12.*1.92e+2-t9.*t12.*t14.*4.0+t11.*t12.*t14+a.*t7.*t12.*t13.*4.8e+1+t8.*t9.*t12.*t14.*6.0-a.*t7.^3.*t12.*t13.*4.8e+1-a.*t7.*t8.*t12.*t13.*1.44e+2;

        elseif (idx == 0) && (idy == 0) && (idt == 5)
            t2 = a.^2;
            t3 = a.*t.*4.0;
            t4 = t-1.0;
            t5 = a.*t4.*4.0;
            t6 = t3.*t4;
            t7 = cos(t6);
            t8 = sin(t6);
            t12 = t3+t5;
            t9 = t7.^2;
            t10 = t7.^3;
            t11 = t8.^2;
            t13 = -t8;
            t15 = t12.^3;
            t16 = t12.^5;
            t14 = exp(t13);
            out = -t7.^5.*t14.*t16-t7.*t14.*t16+t10.*t14.*t16.*1.0e+1-a.*t8.*t14.*t15.*8.0e+1-a.*t9.*t14.*t15.*3.2e+2+a.*t11.*t14.*t15.*2.4e+2+t2.*t7.*t12.*t14.*9.6e+2-t2.*t10.*t12.*t14.*9.6e+2+t7.*t8.*t14.*t16.*1.5e+1-t7.*t11.*t14.*t16.*1.5e+1-t8.*t10.*t14.*t16.*1.0e+1+a.*t9.^2.*t14.*t15.*8.0e+1+a.*t8.*t9.*t14.*t15.*4.8e+2-t2.*t7.*t8.*t12.*t14.*2.88e+3;
        elseif (idx == 0) && (idy == 0) && (idt == 6)
            t2 = a.^2;
            t3 = a.^3;
            t4 = a.*t.*4.0;
            t5 = t-1.0;
            t6 = a.*t5.*4.0;
            t7 = t4.*t5;
            t8 = cos(t7);
            t9 = sin(t7);
            t14 = t4+t6;
            t10 = t8.^2;
            t11 = t8.^3;
            t13 = t9.^2;
            t15 = -t9;
            t17 = t14.^2;
            t12 = t10.^2;
            t16 = exp(t15);
            t18 = t17.^2;
            t19 = t17.^3;
            et1 = t9.^3.*t16.*t19.*1.5e+1+t10.^3.*t16.*t19+t3.*t8.*t16.*7.68e+3-t3.*t11.*t16.*7.68e+3+t9.*t16.*t19+t10.*t16.*t19.*1.6e+1-t12.*t16.*t19.*2.0e+1-t13.*t16.*t19.*1.5e+1-a.*t8.*t16.*t18.*1.2e+2+a.*t11.*t16.*t18.*1.2e+3-t3.*t8.*t9.*t16.*2.304e+4-t2.*t9.*t16.*t17.*2.88e+3-t2.*t10.*t16.*t17.*1.152e+4+t2.*t12.*t16.*t17.*2.88e+3+t2.*t13.*t16.*t17.*8.64e+3-t9.*t10.*t16.*t19.*7.5e+1+t9.*t12.*t16.*t19.*1.5e+1+t10.*t13.*t16.*t19.*4.5e+1-a.*t8.^5.*t16.*t18.*1.2e+2+a.*t8.*t9.*t16.*t18.*1.8e+3-a.*t9.*t11.*t16.*t18.*1.2e+3-a.*t8.*t13.*t16.*t18.*1.8e+3;
            et2 = t2.*t9.*t10.*t16.*t17.*1.728e+4;
            out = et1+et2;

        elseif (idx == 0) && (idy == 0) && (idt == 7)
            t2 = a.^2;
            t3 = a.^3;
            t4 = a.*t.*4.0;
            t5 = t-1.0;
            t6 = a.*t5.*4.0;
            t7 = t4.*t5;
            t8 = cos(t7);
            t9 = sin(t7);
            t16 = t4+t6;
            t10 = t8.^2;
            t11 = t8.^3;
            t13 = t8.^5;
            t14 = t9.^2;
            t15 = t9.^3;
            t17 = -t9;
            t19 = t16.^3;
            t20 = t16.^5;
            t21 = t16.^7;
            t12 = t10.^2;
            t18 = exp(t17);
            et1 = -t8.^7.*t18.*t21+t8.*t18.*t21-t11.*t18.*t21.*9.1e+1+t13.*t18.*t21.*3.5e+1+a.*t9.*t18.*t20.*1.68e+2+a.*t10.*t18.*t20.*2.688e+3-a.*t12.*t18.*t20.*3.36e+3-a.*t14.*t18.*t20.*2.52e+3+a.*t15.*t18.*t20.*2.52e+3-t3.*t9.*t16.*t18.*5.376e+4-t2.*t8.*t18.*t19.*6.72e+3-t3.*t10.*t16.*t18.*2.1504e+5+t3.*t12.*t16.*t18.*5.376e+4+t2.*t11.*t18.*t19.*6.72e+4+t3.*t14.*t16.*t18.*1.6128e+5-t2.*t13.*t18.*t19.*6.72e+3-t8.*t9.*t18.*t21.*6.3e+1+t9.*t11.*t18.*t21.*2.45e+2+t8.*t14.*t18.*t21.*2.1e+2-t9.*t13.*t18.*t21.*2.1e+1-t8.*t15.*t18.*t21.*1.05e+2-t11.*t14.*t18.*t21.*1.05e+2;
            et2 = a.*t10.^3.*t18.*t20.*1.68e+2-a.*t9.*t10.*t18.*t20.*1.26e+4+a.*t9.*t12.*t18.*t20.*2.52e+3+a.*t10.*t14.*t18.*t20.*7.56e+3+t2.*t8.*t9.*t18.*t19.*1.008e+5+t3.*t9.*t10.*t16.*t18.*3.2256e+5-t2.*t9.*t11.*t18.*t19.*6.72e+4-t2.*t8.*t14.*t18.*t19.*1.008e+5;
            out = et1+et2;

        end
    elseif ifun == 9
        k = 1;
        a = x*y*k;
        if (idx == 0) && (idy == 0) && (idt == 0)
            out1 = exp(sin(a.*sin(t)));
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            t2 = sin(t);
            t3 = a.*t2;
            out1 = a.*exp(sin(t3)).*cos(t).*cos(t3);
        elseif (idx == 2) && (idy == 0) && (idt == 0)
            t2 = sin(t);
            t3 = k.^2;
            t4 = y.^2;
            t5 = t2.^2;
            t6 = k.*t2.*x.*y;
            t7 = sin(t6);
            t8 = exp(t7);
            out1 = t3.*t4.*t5.*t8.*cos(t6).^2-t3.*t4.*t5.*t7.*t8;
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            t2 = sin(t);
            t3 = k.^2;
            t4 = x.^2;
            t5 = t2.^2;
            t6 = k.*t2.*x.*y;
            t7 = sin(t6);
            t8 = exp(t7);
            out1 = t3.*t4.*t5.*t8.*cos(t6).^2-t3.*t4.*t5.*t7.*t8;
        elseif (idx == 2) && (idy == 0) && (idt == 1)
            t2 = cos(t);
            t3 = sin(t);
            t4 = k.^2;
            t5 = k.^3;
            t6 = y.^2;
            t7 = y.^3;
            t8 = t3.^2;
            t9 = k.*t3.*x.*y;
            t10 = cos(t9);
            t11 = sin(t9);
            t12 = exp(t11);
            out1 = t2.*t3.*t4.*t6.*t11.*t12.*-2.0+t2.*t3.*t4.*t6.*t10.^2.*t12.*2.0-t2.*t5.*t7.*t8.*t10.*t12.*x+t2.*t5.*t7.*t8.*t10.^3.*t12.*x-t2.*t5.*t7.*t8.*t10.*t11.*t12.*x.*3.0;

        elseif (idx == 0) && (idy == 2) && (idt == 1)
            t2 = cos(t);
            t3 = sin(t);
            t4 = k.^2;
            t5 = k.^3;
            t6 = x.^2;
            t7 = x.^3;
            t8 = t3.^2;
            t9 = k.*t3.*x.*y;
            t10 = cos(t9);
            t11 = sin(t9);
            t12 = exp(t11);
            out1 = t2.*t3.*t4.*t6.*t11.*t12.*-2.0+t2.*t3.*t4.*t6.*t10.^2.*t12.*2.0-t2.*t5.*t7.*t8.*t10.*t12.*y+t2.*t5.*t7.*t8.*t10.^3.*t12.*y-t2.*t5.*t7.*t8.*t10.*t11.*t12.*y.*3.0;
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = t2.^2;
            t6 = a.*t3;
            t7 = cos(t6);
            t8 = sin(t6);
            t9 = exp(t8);
            out1 = -t6.*t7.*t9-t4.*t5.*t8.*t9+t4.*t5.*t7.^2.*t9;
        elseif (idx == 0) && (idy == 0) && (idt == 3)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = a.^3;
            t6 = t2.^3;
            t7 = a.*t3;
            t8 = cos(t7);
            t9 = sin(t7);
            t10 = exp(t9);
            out1 = -a.*t2.*t8.*t10-t5.*t6.*t8.*t10+t5.*t6.*t8.^3.*t10-t2.*t3.*t4.*t8.^2.*t10.*3.0+t2.*t3.*t4.*t9.*t10.*3.0-t5.*t6.*t8.*t9.*t10.*3.0;
        elseif (idx == 0) && (idy == 0) && (idt == 4)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = a.^3;
            t6 = t4.^2;
            t7 = t2.^2;
            t9 = t3.^2;
            t10 = a.*t3;
            t8 = t7.^2;
            t11 = cos(t10);
            t12 = sin(t10);
            t13 = exp(t12);
            t14 = t11.^2;
            out1 = t10.*t11.*t13+t4.*t7.*t12.*t13.*4.0-t4.*t7.*t13.*t14.*4.0-t4.*t9.*t12.*t13.*3.0+t6.*t8.*t12.*t13+t4.*t9.*t13.*t14.*3.0-t6.*t8.*t13.*t14.*4.0+t6.*t8.*t12.^2.*t13.*3.0+t6.*t8.*t13.*t14.^2-t3.*t5.*t7.*t11.^3.*t13.*6.0+t3.*t5.*t7.*t11.*t13.*6.0-t6.*t8.*t12.*t13.*t14.*6.0+t3.*t5.*t7.*t11.*t12.*t13.*1.8e+1;
        elseif (idx == 0) && (idy == 0) && (idt == 5)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = a.^3;
            t7 = a.^5;
            t6 = t4.^2;
            t8 = t2.^3;
            t9 = t2.^5;
            t10 = t3.^2;
            t11 = a.*t3;
            t12 = cos(t11);
            t13 = sin(t11);
            t14 = exp(t13);
            t15 = t12.^2;
            t16 = t12.^3;
            t17 = t13.^2;
            out1 = a.*t2.*t12.*t14+t5.*t8.*t12.*t14.*1.0e+1+t7.*t9.*t12.*t14-t5.*t8.*t14.*t16.*1.0e+1-t7.*t9.*t14.*t16.*1.0e+1+t7.*t9.*t12.^5.*t14-t3.*t6.*t8.*t14.*t15.^2.*1.0e+1-t2.*t3.*t4.*t13.*t14.*1.5e+1+t2.*t3.*t4.*t14.*t15.*1.5e+1-t2.*t5.*t10.*t12.*t14.*1.5e+1-t3.*t6.*t8.*t13.*t14.*1.0e+1+t3.*t6.*t8.*t14.*t15.*4.0e+1+t2.*t5.*t10.*t14.*t16.*1.5e+1-t3.*t6.*t8.*t14.*t17.*3.0e+1+t5.*t8.*t12.*t13.*t14.*3.0e+1+t7.*t9.*t12.*t13.*t14.*1.5e+1+t7.*t9.*t12.*t14.*t17.*1.5e+1-t7.*t9.*t13.*t14.*t16.*1.0e+1-t2.*t5.*t10.*t12.*t13.*t14.*4.5e+1+t3.*t6.*t8.*t13.*t14.*t15.*6.0e+1;
        elseif (idx == 0) && (idy == 0) && (idt == 6)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = a.^3;
            t7 = a.^5;
            t6 = t4.^2;
            t8 = t4.^3;
            t9 = t2.^2;
            t12 = t3.^2;
            t13 = t3.^3;
            t14 = a.*t3;
            t10 = t9.^2;
            t11 = t9.^3;
            t15 = cos(t14);
            t16 = sin(t14);
            t17 = exp(t16);
            t18 = t15.^2;
            t19 = t15.^3;
            t21 = t16.^2;
            t20 = t18.^2;
            et1 = -t14.*t15.*t17-t4.*t9.*t16.*t17.*1.6e+1+t4.*t9.*t17.*t18.*1.6e+1+t4.*t12.*t16.*t17.*1.5e+1-t6.*t10.*t16.*t17.*2.0e+1+t5.*t13.*t15.*t17.*1.5e+1-t4.*t12.*t17.*t18.*1.5e+1+t6.*t10.*t17.*t18.*8.0e+1-t8.*t11.*t16.*t17-t6.*t10.*t17.*t20.*2.0e+1-t5.*t13.*t17.*t19.*1.5e+1-t6.*t10.*t17.*t21.*6.0e+1+t8.*t11.*t17.*t18.*1.6e+1-t8.*t11.*t17.*t20.*2.0e+1-t8.*t11.*t17.*t21.*1.5e+1-t8.*t11.*t16.^3.*t17.*1.5e+1+t8.*t11.*t17.*t18.^3-t3.*t7.*t10.*t15.^5.*t17.*1.5e+1-t3.*t5.*t9.*t15.*t17.*7.5e+1-t3.*t7.*t10.*t15.*t17.*1.5e+1+t3.*t5.*t9.*t17.*t19.*7.5e+1+t3.*t7.*t10.*t17.*t19.*1.5e+2+t6.*t9.*t12.*t16.*t17.*4.5e+1-t6.*t9.*t12.*t17.*t18.*1.8e+2;
            et2 = t6.*t9.*t12.*t17.*t20.*4.5e+1+t6.*t9.*t12.*t17.*t21.*1.35e+2+t5.*t13.*t15.*t16.*t17.*4.5e+1+t6.*t10.*t16.*t17.*t18.*1.2e+2+t8.*t11.*t16.*t17.*t18.*7.5e+1-t8.*t11.*t16.*t17.*t20.*1.5e+1+t8.*t11.*t17.*t18.*t21.*4.5e+1-t3.*t5.*t9.*t15.*t16.*t17.*2.25e+2-t3.*t7.*t10.*t15.*t16.*t17.*2.25e+2+t3.*t7.*t10.*t16.*t17.*t19.*1.5e+2-t3.*t7.*t10.*t15.*t17.*t21.*2.25e+2-t6.*t9.*t12.*t16.*t17.*t18.*2.7e+2;
            out1 = et1+et2;
        elseif (idx == 0) && (idy == 0) && (idt == 7)
            t2 = cos(t);
            t3 = sin(t);
            t4 = a.^2;
            t5 = a.^3;
            t7 = a.^5;
            t9 = a.^7;
            t6 = t4.^2;
            t8 = t4.^3;
            t10 = t2.^3;
            t11 = t2.^5;
            t12 = t2.^7;
            t13 = t3.^2;
            t14 = t3.^3;
            t15 = a.*t3;
            t16 = cos(t15);
            t17 = sin(t15);
            t18 = exp(t17);
            t19 = t16.^2;
            t20 = t16.^3;
            t22 = t16.^5;
            t23 = t17.^2;
            t24 = t17.^3;
            t21 = t19.^2;
            et1 = -a.*t2.*t16.*t18-t5.*t10.*t16.*t18.*9.1e+1-t7.*t11.*t16.*t18.*3.5e+1+t5.*t10.*t18.*t20.*9.1e+1-t9.*t12.*t16.*t18+t7.*t11.*t18.*t20.*3.5e+2-t7.*t11.*t18.*t22.*3.5e+1+t9.*t12.*t18.*t20.*9.1e+1-t9.*t12.*t18.*t22.*3.5e+1+t9.*t12.*t16.^7.*t18-t3.*t8.*t11.*t18.*t19.^3.*2.1e+1+t2.*t3.*t4.*t17.*t18.*6.3e+1-t2.*t3.*t4.*t18.*t19.*6.3e+1+t2.*t5.*t13.*t16.*t18.*2.1e+2+t3.*t6.*t10.*t17.*t18.*2.45e+2-t3.*t6.*t10.*t18.*t19.*9.8e+2-t2.*t6.*t14.*t17.*t18.*1.05e+2+t3.*t8.*t11.*t17.*t18.*2.1e+1-t2.*t5.*t13.*t18.*t20.*2.1e+2+t3.*t6.*t10.*t18.*t21.*2.45e+2+t2.*t6.*t14.*t18.*t19.*4.2e+2-t3.*t8.*t11.*t18.*t19.*3.36e+2;
            et2 = t3.*t6.*t10.*t18.*t23.*7.35e+2-t2.*t6.*t14.*t18.*t21.*1.05e+2+t3.*t8.*t11.*t18.*t21.*4.2e+2-t2.*t6.*t14.*t18.*t23.*3.15e+2+t3.*t8.*t11.*t18.*t23.*3.15e+2+t3.*t8.*t11.*t18.*t24.*3.15e+2+t7.*t10.*t13.*t16.*t18.*1.05e+2-t5.*t10.*t16.*t17.*t18.*2.73e+2-t7.*t10.*t13.*t18.*t20.*1.05e+3-t7.*t11.*t16.*t17.*t18.*5.25e+2+t7.*t10.*t13.*t18.*t22.*1.05e+2-t9.*t12.*t16.*t17.*t18.*6.3e+1+t7.*t11.*t17.*t18.*t20.*3.5e+2-t7.*t11.*t16.*t18.*t23.*5.25e+2+t9.*t12.*t17.*t18.*t20.*2.45e+2-t9.*t12.*t16.*t18.*t23.*2.1e+2-t9.*t12.*t17.*t18.*t22.*2.1e+1-t9.*t12.*t16.*t18.*t24.*1.05e+2+t9.*t12.*t18.*t20.*t23.*1.05e+2+t2.*t5.*t13.*t16.*t17.*t18.*6.3e+2;
            et3 = t3.*t6.*t10.*t17.*t18.*t19.*-1.47e+3+t2.*t6.*t14.*t17.*t18.*t19.*6.3e+2-t3.*t8.*t11.*t17.*t18.*t19.*1.575e+3+t3.*t8.*t11.*t17.*t18.*t21.*3.15e+2+t7.*t10.*t13.*t16.*t17.*t18.*1.575e+3-t3.*t8.*t11.*t18.*t19.*t23.*9.45e+2-t7.*t10.*t13.*t17.*t18.*t20.*1.05e+3+t7.*t10.*t13.*t16.*t18.*t23.*1.575e+3;
            out1 = et1+et2+et3;
        end
        out = out1;
    elseif ifun == 10
        if (idx == 0) && (idy == 0) && (idt == 0)
            out1 = exp(sin(t+x)).*exp(cos(t-y))+t.*x.*y;
        elseif (idx == 0) && (idy == 0) && (idt == 1)
            t2 = t+x;
            t4 = -y;
            t3 = sin(t2);
            t6 = t+t4;
            t5 = exp(t3);
            t7 = cos(t6);
            t8 = exp(t7);
            out1 = x.*y+t5.*t8.*cos(t2)-t5.*t8.*sin(t6);
        elseif (idx == 0) && (idy == 0) && (idt == 2)
            t2 = t+x;
            t5 = -y;
            t3 = cos(t2);
            t4 = sin(t2);
            t7 = t+t5;
            t6 = exp(t4);
            t8 = cos(t7);
            t9 = sin(t7);
            t10 = exp(t8);
            out1 = t3.^2.*t6.*t10+t6.*t9.^2.*t10-t4.*t6.*t10-t6.*t8.*t10-t3.*t6.*t9.*t10.*2.0;


        elseif (idx == 2) && (idy == 0) && (idt == 0)
            t2 = t+x;
            t4 = -y;
            t3 = sin(t2);
            t6 = t+t4;
            t5 = exp(t3);
            t7 = cos(t6);
            t8 = exp(t7);
            out1 = t5.*t8.*cos(t2).^2-t3.*t5.*t8;
        elseif (idx == 2) && (idy == 0) && (idt == 1)

            t2 = t+x;
            t5 = -y;
            t3 = cos(t2);
            t4 = sin(t2);
            t7 = t+t5;
            t6 = exp(t4);
            t8 = cos(t7);
            t9 = sin(t7);
            t10 = exp(t8);
            out1 = t3.^3.*t6.*t10-t3.*t6.*t10-t3.*t4.*t6.*t10.*3.0+t4.*t6.*t9.*t10-t3.^2.*t6.*t9.*t10;
        elseif (idx == 0) && (idy == 2) && (idt == 0)
            t2 = t+x;
            t4 = -y;
            t3 = sin(t2);
            t6 = t+t4;
            t5 = exp(t3);
            t7 = cos(t6);
            t8 = exp(t7);
            out1 = t5.*t8.*sin(t6).^2-t5.*t7.*t8;
        elseif (idx == 0) && (idy == 2) && (idt == 1)
            t2 = t+x;
            t5 = -y;
            t3 = cos(t2);
            t4 = sin(t2);
            t7 = t+t5;
            t6 = exp(t4);
            t8 = cos(t7);
            t9 = sin(t7);
            t10 = exp(t8);
            out1 = -t6.*t9.^3.*t10+t6.*t9.*t10-t3.*t6.*t8.*t10+t6.*t8.*t9.*t10.*3.0+t3.*t6.*t9.^2.*t10;
        end
        out = out1;
    else
        display('Function not implemented')
        return
    end

end