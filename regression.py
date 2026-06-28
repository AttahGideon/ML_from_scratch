Physics = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
History = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]

mean_phy = sum(Physics) / len(Physics)
mean_his = sum(History) / len(History)

def compute_m(Physics, History, mean_phy, mean_his):
    numerator = sum((p - mean_phy) * (h - mean_his) for p, h in zip(Physics, History))
    
    denominator = sum((p - mean_phy)**2 for p in Physics)
    
    m = numerator / denominator
    return m
    
w = compute_m(Physics, History, mean_phy, mean_his)

b = mean_his - w * mean_phy
x = 10

def compute_reg(w, x, b):
    return w*10 + b
    
work = compute_reg(w, x, b)
print(float(work))
