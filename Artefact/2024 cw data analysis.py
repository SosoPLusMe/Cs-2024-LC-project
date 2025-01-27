import statistics
import matplotlib.pyplot as plt
import numpy as np

sleepFile = open("sleepdata.txt", "r")
sleepRec = sleepFile.readline() 
sleepRec = sleepFile.readline()

awokenList = []
sleepdList = []
stressList = []
qualityList = []
timeList = []
noiseavgList = []


while sleepRec != "":
    field = sleepRec.split(",")
    
    # store each field in a variable + Validation
    timesup = int(field[0])
    sleepdur = float(field[1])
    stresslvl = int(field[2])
    qsleep = int(field[3])
    runtime = float(field[4])
    noiseavg = float(field[5])


    # etc., for remaining fields
    
    # append these variables to the appropriate list
    awokenList.append(timesup)
    sleepdList.append(sleepdur)
    stressList.append(stresslvl)
    qualityList.append(qsleep)
    timeList.append(runtime)
    noiseavgList.append(noiseavg)
    
    # read the next line of the file
    sleepRec = sleepFile.readline()
    

sleepFile.close()

print(' ')

print('This is the list of the amount times the user awoke',awokenList)
print(' ')
print('This is the list of the durations the user was asleep',sleepdList)
print(' ')
print('This is the list of the stress levels of the user',stressList)
print(' ')
print('This is the list of the sleep quality of according to the user',qualityList)
print(' ')
print('This is the list of the duration the artefact was turned on',timeList)
print(' ')
print('This is the list of the average noise levels',noiseavgList)
print(' ')

plt.bar(awokenList,noiseavgList, color ='maroon', width = 0.4)
plt.title('Noise Graph')
plt.xlabel('Times Awoken')
plt.ylabel('NOISE LEVEL')
plt.show()



noiseList = [0,18,32,41,23,13,42,21,23]
timList = [0,10,20,30,40,50,60,70,80]

'''
x = np.array(awokenList)
y = np.array(noiseavgList)

coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

plt.xlim(0, 10)
plt.ylim(0, 40)
awokenList.append(xlims[1])
noiseavgList.append(np.polyval(coef, xlims[1]))


plt.scatter(awokenList, noiseavgList)
plt.title('Noise Graph')
plt.xlabel('TIMES AWOKEN')
plt.ylabel('NOISE LEVEL')
plt.show()
'''

x = awokenList
y = noiseavgList

p1 = np.polyfit(x, y, 1)

plt.scatter(x, y ,marker='+')
xlims = plt.xlim()
ylims = plt.ylim()
#x.insert(0, xlims[0])
#y.insert(0, np.polyval(p1, xlims[0]))
x.append(xlims[1]+9)
y.append(ylims[1]+9)
#y.append(np.polyval(p1, xlims[0]))
plt.plot(x, np.polyval(p1,x), 'r-', linewidth = 1)
plt.xlim(1,10)
plt.ylim(0, 50)
plt.title('Noise Graph')
plt.xlabel('TIMES AWOKEN')
plt.ylabel('NOISE LEVEL')
plt.show()

print("The slope is", p1[0])
print("The intercept is", p1[1])
print("The equation of the line is y="+str(round(p1[0],2))+"x+"+str(round(p1[1],2)))

predUP = int(input("Enter the amount of times you will awake: "))
y=p1[0]*(predUP)+(p1[1])

print ("The average noise level you will have if you awake that amaount of times is", round(y,2))

def advice(y):
    if y < 10:
        return "This is a healthy amount of noise level to have whilst you sleep"
    elif y > 10 and y < 30:
        return "You may awake multiple times throughout the night but the noise level is not a big concern."
    elif y > 30:
        return "The noise level has started to negatively impact your sleep this may impact your mood negatively too, using rugs, carpets, and soft furnishings can keep the noise level down"
    else:
        return "The noise level is too high and this will negatively impact your sleep and your mood you may need to use wooden shutters or noise dampening curtains if the noise comes from the outside and use bookshelf and furniture as a wall divider "

adv = advice(y)
print(adv)
