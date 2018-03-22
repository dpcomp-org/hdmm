
sens = @(A) max(sum(abs(A)));

error = @(W,A) sens(A) * sqrt(trace((W' * W) * pinv(A' * A)));

for n = [32, 64, 128, 256, 512, 1024]
    W = tril(ones(n,n));
    tic;
    [~,A] = LowRankDP(W);
    time = toc;
    err1 = error(W, A);
    err2 = error(W, eye(size(W,2)));
    fprintf('%d, %.2f, %.2f, %.2f \n', n, err1, err2, time);
end
    