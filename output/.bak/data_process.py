import re, math


invTs = {'0':[], '1':[]}
losses = {'0':[], '1':[]}

for l in open('loss_for_metropolis'):
    l = l.strip()

    #m = re.match(r'Epoch (\d+).*InvT: (\d.\d\dE+\d\d) Loss: (\d.\d+) ', l)
    m = re.match(r'Epoch (\d+) Chain (\d+).*InvT: (\d\.\d\dE\+\d\d)  Loss: (\d\.\d+) ', l)
    if m:
        epoch = m.group(1)
        chain = m.group(2)
        invT = m.group(3)
        loss = m.group(4)
        invTs[chain].append(float(invT))
        losses[chain].append(float(loss))

accepts = 0
for i in range(1000):
    choice = math.exp(min(0, (invTs['1'][i] - invTs['0'][i]) * (losses['1'][i] - losses['0'][i] * 1.05)))
    accepts += choice
    if choice > 0:
        print(i, choice)
print("total accepts", accepts)
