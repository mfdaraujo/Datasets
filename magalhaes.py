# Magalhães
media = np.mean(galeria,axis=0)
std = np.std(galeria,axis=0)
median = np.median(galeria, axis=0)
print(f'média: {media}\nstd: {std}\nmediana: {median}')

n = len(amostra)
score = np.array([])

for i in range(n):
    tlp = amostra[i]
    ft_media = media[i]
    ft_std = std[i]
    ft_mediana = median[i]
    low_interval = np.min([ft_media,ft_mediana]) * (0.95 - ft_std/ft_media)
    high_interval = np.max([ft_media,ft_mediana]) * (1.05 + ft_std/ft_media)
    print(f'low_interval: {low_interval}\ntlp: {tlp}\nhigh_interval: {high_interval}')
    if (tlp >= low_interval) & (tlp <= high_interval):
        if i == 0:
            score = np.append(score,1)
        else:
            if (score[i-1] == 1)|(score[i-1] == 1.5):
                score = np.append(score,1.5)
            else:
                score = np.append(score, 1)
    else:
        score = np.append(score, 0)

score = 1 - score.sum()/(1.5*n-0.5)
