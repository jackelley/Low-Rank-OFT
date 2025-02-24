function void = mat2order_table(A)
    
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
            str = sprintf('%3.3f',A(i,j));
            if (j < n) 
                sss = [sss [' ' str(1:4)  ' & ' ]];
            else
                sss = [sss [' ' str(1:4)  ' \\' ]];
            end
        end    
        disp(sss)
    end
        disp(' \end{tabular}')
    disp(' \caption{This is the caption} ')
    disp(' \end{center} ')
    disp(' \end{table} ')

end