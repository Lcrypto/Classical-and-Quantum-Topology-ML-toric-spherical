%Copyright(c) 2023, Usatyuk Vasiliy 
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are met :
%*Redistributions of source code must retain the above copyright
%notice, this list of conditions and the following disclaimer.
%* Redistributions in binary form must reproduce the above copyright
%notice, this list of conditions and the following disclaimer in the
%documentation and / or other materials provided with the distribution.
%* Neither the name of the <organization> nor the
%names of its contributors may be used to endorse or promote products
%derived from this software without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED.IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
%DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Core contribution LDPC based Tensor Factorization for future use to
% simplify Complex Transformer Deep Neural Neutwork which contain billions parameters
% based on the Matlab platform from https://github.com/RuslanKhalitov/SparseFactorization
%Ruslan Khalitov, Tong Yu, Lei Cheng, Zhirong Yang,
%Sparse factorization of square matrices with application to neural attention modeling,
%Neural Networks,Volume 152,2022, Pages 160-168,

%datalist = readmatrix('datalist.csv', 'OutputType', 'char');
datalist = readmatrix('datalist_full.csv', 'OutputType', 'char');
filenames = datalist(:,1);
categories = datalist(:,2);
nd = length(filenames);



max_iter = 100000; % fast check
Check_LDPC=1; % if 1 use max ACE nonstructured LDPC 
%if 0 original method decompose matrix chord (cage like) 
maxDv=4; %maximal column weight
irregular_profile=-1; % if -1 use similar weigh as cage it important for fair compare
% Frobenius NORM with TSVDs, 1 irregular,regular with maxDv weight
start_from=49;
nd=49;






for di=start_from:nd
    fprintf('============= di=%d/%d =====================\n', di, nd);
    fprintf(categories{di});
    fprintf("\n",filenames{di});
    
    A = load_square_matrix(categories{di}, filenames{di});
    res = sf_appr_test(A, max_iter,Check_LDPC,irregular_profile,maxDv);
    fres = sprintf('results/sf_appr_test_di%d.mat', di);
    save(fres, 'res', 'filenames', 'categories', 'max_iter');
end




