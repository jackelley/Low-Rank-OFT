function void = mat2err_table(A)
    
    [m,n] = size(A);
    disp(' \begin{table}[] ')
    disp(' \begin{center} ')
    
    sss = '\begin{tabular}{| ';
    for j = 1:n
        sss = [sss ' c '];
    end
    sss = [sss ' |} '];
    disp(sss)
    for i = 1:m
        
        sss = '';
        for j = 1:n
            str = sprintf('%0.3e',A(i,j));
            expo = log10(A(i,j));
            if (expo < 0)
                str_expo = sprintf('%d',floor(expo));
            else
                str_expo = sprintf('%d',ceil(expo));
            end
            if (j < n) 
                sss = [sss [' ' str(1:4) '(' str_expo ') & ']];
            else
                sss = [sss [' ' str(1:4) '(' str_expo ') \\']];
            end
        end    
        disp(sss)
    end
    disp(' \end{tabular}')
    disp(' \caption{This is the caption} ')
    disp(' \end{center} ')
    disp(' \end{table} ')
    
    


end