Physics = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
History = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]

phy_mean = sum(Physics)/len(Physics)
his_mean = sum(History)/len(History)

def calc_correlation(Physics, History, phy_mean, his_mean):
    numerator = sum((p - phy_mean) * (h - his_mean) for p, h in zip(Physics, History))
    
    sum_sq_ph = sum((p - phy_mean)**2 for p in Physics)
    sum_sq_his = sum((h - his_mean)**2 for h in History)
    
    denominator = ( sum_sq_ph * sum_sq_his)**0.5
    r = numerator/denominator
    return r
    
correlation = calc_correlation(Physics, History, phy_mean, his_mean)
    
print(round(correlation, 3))
