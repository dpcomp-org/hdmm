import workload
import templates
import numpy as np
from experiments.utility import wavelet, hb, identity, greedyH


W1 = workload.WidthKRange(1024, 32)
W2 = workload.Prefix(1024)

eye = np.eye(1024)
wav = wavelet(1024).toarray()
hier = hb(1024).toarray()
greedy1 = greedyH(W1.WtW).toarray()
greedy2 = greedyH(W2.WtW).toarray()

print W1.rootmse(eye), W1.rootmse(wav), W1.rootmse(hier), W1.rootmse(greedy1)
print W2.rootmse(eye), W2.rootmse(wav), W2.rootmse(hier), W2.rootmse(greedy2)

pid = templates.PIdentity(1024//16, 1024)
pid.restart_optimize(W1, 5)
print W1.rootmse(pid.A)

pid = templates.PIdentity(1024//16, 1024)
pid.optimize(W2)
print W2.rootmse(pid.A)


