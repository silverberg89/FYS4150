import numpy as np

# Players
P = ['Rune', 'Marius', 'Marcus', 'Samuel']

# Structure
Penalty = 100
MAX = 1
Players = len(P)

# Calculation
N = 38
door = 0
Sum = np.zeros(N+1)
Pen = np.zeros(Players)
ID = np.zeros(Players)
for i in range(1,N+1):
    Temp_sum = 0
    for j in range(Players):
        Pen[j] = Penalty*(j/(Players-1))
        Temp_sum += Pen[j]
        if door == 0:
            print('Position penalty:',j,int(Pen[j]),'NOK')
    door = 1
    Sum[i] = Temp_sum+Sum[i-1]
    #print('Pott etter omgang',i,':', int(Sum[i]),'NOK')
    Sum_best = Sum[-1]
    
Roof = Sum[-1]/(Players-MAX)
####################################### One roof worst case
Roof_break_omg = int(Roof/Pen[-1])
print('\nMax payment:',int(Roof),'NOK','\nEarliest omg one roof case:',Roof_break_omg)
Sum_roof = np.copy(Sum)
Sum_roof[Roof_break_omg+1:N+1] = 0
for i in range(Roof_break_omg+1,N+1):
    Sum_roof[i] = Sum_roof[i-1]
    for k in range(Players-1):
        Sum_roof[i] += Pen[k]
    #print('Minimum pott etter omgang',i,':', int(Sum_roof[i]),'NOK')
    Sum_one_roof_worst = Sum_roof[-1]
####################################### Two roof worst case
Roof_break_omg = int(Roof*2/(Pen[-1]+Pen[-2]))
print('Earliest omg two roof case:',Roof_break_omg)
Sum_roof = np.copy(Sum_roof)
Sum_roof[Roof_break_omg+1:N+1] = 0
for i in range(Roof_break_omg+1,N+1):
    Sum_roof[i] = Sum_roof[i-1]
    for k in range(Players-2):
        Sum_roof[i] += Pen[k]
    #print('Minimum pott etter omgang',i,':', int(Sum_roof[i]),'NOK')
    Sum_two_roof_worst = Sum_roof[-1]

print('\nBest case:',int(Sum_best),'\nWorst case one roof:',int(Sum_one_roof_worst),'\nWorst case two roofs:',int(Sum_two_roof_worst))
