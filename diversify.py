import numpy as np
import math
from scipy.optimize import minimize

#Case 1 (single currency)
def solvePools(poolData, rho, R, lA, *, PPS = False, debug=False, formatStr = '%s'):
    '''
    Determines how much power should be put into each minimg pool based on their power and fees
    
    Args:
        poolData (List<Tuple<float, float>>): The mining power and fees for each pool.
        rho (float): CARA - constant absolute risk aversion.
        R (float): Reward per block.
        lA (float): Miner's total hashing power.
        debug (bool): Enable debugging messsages. (default: False)
        formatStr (str): Custom format string. (default: '%s')

    Returns:
        numpy array: A vector representing how much power to put towards each mining pool.
                     Returns None if optimization fails
    Example:
    solvePools([(1E+6, 0.03), (1E+5, 0.02),(0,1)], 8e-5, 80000,4e+2,PPS=True)
    '''

    def objective(x, sign= -1.0):
        total = 0
        if PPS:
            for i, (xi, pooli) in enumerate(zip(x, poolData)):
                Li, fi = pooli
                if (i == len(poolData)-2):
                    total = total + xi*(1 - fi)*rho*R
                    #print("pooldata:" + str(poolData))
                    #print("xi:"+str(xi))
                    #print("fi:"+str(fi))
                else:
                    total = total + (xi + Li)*(1 - math.exp((-1)*rho*R*(1-fi)*xi/(xi+Li)))
        else:
            for xi, pooli in zip(x, poolData):
                Li, fi = pooli
                total = total + (xi + Li)*(1 - math.exp((-1)*rho*R*(1-fi)*xi/(xi+Li)))
        #uncomment if needed to scale down
        #total = round((total)**(1),18)
        total *= sign
        return total

    def constraint1(x, sign= -1.0):
        return sign*(sum(x) - lA)

    def constraint2(x, sign= -1.0):
        return (x - 0)

    poolData.append((0,0)) #append solo pool
    n = len(poolData)
    poolData = [list(l) for l in poolData] #convert to list of lists
    minHash = poolData[0][0]
    for m in range (0,n-1): #Find smallest value of pool hashrates except solo pool
        if (poolData[m][0] < minHash):
            minHash = poolData[m][0]
    minHash = lA #(override: normalize to lA)
    for m in range (0,n): #normalize pool hashrates to minHash
        poolData[m][0] = poolData[m][0]/minHash
    lA = lA/minHash #normalize miner hashrate to minHash
    x0 = np.ones(n) * lA/n  # Initial guess
    global guess
    guess = x0 #store initial guess to global variable
    if debug:
        print('Initial Objective: {formatStr}'.format(formatStr=formatStr) % objective(x0))

    #b = (0,lA)
    #bnds = (b,) * n
    con1 = {'type': 'ineq', 'fun': constraint1} 
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = ([con1,con2])
    
    solution = minimize(objective,x0,method='COBYLA',\
                            constraints=cons,options={'maxiter': 50000, 'disp': False, 'tol' : 1e-65, 'catol' : 1e-15})
    x = solution.x
    #scale back to original values
    x = minHash*x 
    lA = minHash*lA

    global curr_num #number of currencies used for graphs
    curr_num = 1

    if debug:
        # show final objective
        print('Final Objective:   {formatStr}'.format(formatStr=formatStr) % objective(x))

        # print solution
        print('Solution')
        for n, xi in enumerate(x):
            if (n<len(poolData)-1):
                print('l_%d    = {formatStr}'.format(formatStr=formatStr) % (n, xi))
            else:
                print('l_solo    = {formatStr}'.format(formatStr=formatStr) % (xi))
        print(    'l_unused = {formatStr}'.format(formatStr=formatStr) % (lA - sum(x)))   

    if solution.success:
        #x = np.append(x,(lA - sum(x))) #append solo mining to end of numpy array output
        return x
    
    return None

#Case 2 (multiple currencies, single PoW algorithm)
def solvePoolsMultiCurr(poolData, rho, lA, *, debug=False, formatStr = '%s'):
    '''
    Determines how much power should be put into each minimg pool based on their power and fees
    
    Args:
        poolData (List<Tuple<float, float, float, float, float,>>): For each pool: mining power, fees, currency block reward Rc, currency block time Dc, currency total hashrate Lc.
        rho (float): CARA - constant absolute risk aversion.
        lA (float): Miner's total hashing power.
        debug (bool): Enable debugging messsages. (default: False)
        formatStr (str): Custom format string. (default: '%s')

    Returns:
        numpy array: A vector representing how much power to put towards each mining pool.
                     Returns None if optimization fails
    Example:
    solvePoolsMultiCurr([(5.750E+18, 0.02,80135,600,50.24e+18),(0.559E+18, 0.02,80135*0.06957,600,3.51e+18),
        (0.452E+18, 0.009,80135,600,50.24e+18)], 8e-5,12.5e+15)
    '''

    def objective(x, sign= -1.0):
        total = 0
        for xi, pooli in zip(x, poolData):
            Li, fi, Ri, Di, Lc = pooli
            total += (xi + Li)/(Di*Lc)*(1 - math.exp((-1)*rho*Ri*(1-fi)*xi/(xi+Li)))
        total *= sign
        return total

    def constraint1(x, sign= -1.0):
        return sign*(sum(x) - lA)
    def constraint2(x, sign= -1.0):
	    return (x - 0)

    C_all = [] #Holds the currency data for each pool
    for y in poolData:
        C_all.append(y[2:5])
    #C_all_tup = [tuple(l) for l in C_all]
    C = list(set(C_all)) #Make the unique currency data list
    for z in C:
        poolData.append((0,0,z[0],z[1],z[2])) #appends unique solo mining currencies
        #to poolData where mining power = 0 and fees = 0
    n = len(poolData)
    poolData = [list(l) for l in poolData] #convert to list of lists
    minHash = poolData[0][0]
    for m in range (0,n-len(C)): #Find smallest value of pool hashrates (exclude solo currencies)
        if (poolData[m][0] < minHash):
            minHash = poolData[m][0]
    minHash = lA #(override: normalize to lA)
    for m in range (0,n): 
        poolData[m][0] = poolData[m][0]/minHash #normalize pool hashrates to minHash
        poolData[m][4] = poolData[m][4]/minHash #normalize currency hashrates to minHash
    lA = lA/minHash #normalize miner hashrate to minHash

    x0 = np.ones(n) * lA/(n) 
    global guess
    guess = x0 #store guess to global variable
    if debug:
        print('Initial Objective: {formatStr}'.format(formatStr=formatStr) % objective(x0))
    #b = (0,lA)
    #bnds = (b,) * n
    con1 = {'type': 'ineq', 'fun': constraint1} 
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = ([con1,con2])
    solution = minimize(objective,x0,method='COBYLA',\
                            constraints=cons,options={'maxiter': 50000, 'disp': False, 'tol' : 1e-65, 'catol' : 1e-15})	
    x = solution.x
    #scale back to original values
    x = minHash*x 
    lA = minHash*lA

    global curr_num
    curr_num = len(C)

    if debug:
        # show final objective
        print('Final Objective:   {formatStr}'.format(formatStr=formatStr) % objective(x))

        # print solution
        print('Solution')
        curr_ctr = 1
        for n, xi in enumerate(x):
            if (n<len(poolData)-len(C)):
                print('l_%d    = {formatStr}'.format(formatStr=formatStr) % (n, xi))
            else:
                print('Currency',curr_ctr, 'with data:', C[curr_ctr-1])
                print('l_c%d    = {formatStr}'.format(formatStr=formatStr) % (curr_ctr, xi))
                curr_ctr = curr_ctr + 1
        print(    'l_unused = {formatStr}'.format(formatStr=formatStr) % (lA - sum(x)))   

    if solution.success:
        return x
    
    return None

#Case 3 (multiple currencies, multiple PoW algorithm)
def solvePoolsMultiAlg(poolData, rho, *, debug=False, formatStr = '%s'):
    '''
    Determines how much power should be put into each minimg pool based on their power and fees
    This works for multiple currencies under different PoW algorithms

    Args:
        poolData (List<Tuple<float, float, float, float, float, float>>): For each pool: mining power, fees, currency block reward Rc, currency block time Dc, currency total hashrate Lc, maximum hashrate for algorithm a la.
        rho (float): CARA - constant absolute risk aversion.
        debug (bool): Enable debugging messsages. (default: False)
        formatStr (str): Custom format string. (default: '%s')

    Returns:
        numpy array: A vector representing how much power to put towards each mining pool.
                     Returns None if optimization fails
    Example:
    solvePoolsMultiAlg([(68E+12, 0.01,1026,0.25,278e+12,26e+6), (37E+6, 0.01,375,2,422e+6,730)], 8e-5)
    '''

    def objective(x, sign= -1.0):
        total = 0
        A = dict.fromkeys(set(A_all),0) #reset all dictionary values
        for xi, pooli in zip(x, poolData):
            Li, fi, Ri, Di, Lc, ai = pooli
            total += (xi + Li)/(Di*Lc)*(1 - math.exp((-1)*rho*Ri*(1-fi)*xi/(xi+Li)))
            A[ai] += xi
        total *= sign
        return total

    def constraint1(x, sign= -1.0):
        sumx = 0
        j=0
        for i in x:
            sumx += x[j]/poolData[j][5]
            j +=1
        return sign*(sumx - 1)
    def constraint2(x, sign= -1.0):
        return (x - 0)
    '''
    A_all = [] #Holds the algorithm maximum hashrate for each pool
    for v in poolData:
        A_all.append(v[5]) #Read data from function input
    A_unique = dict.fromkeys(set(A_all),0) #Make the algorithm hashrate dictionary
    for alg in poolData:
        A_unique[alg[5]] += 1 #Populate dictionary with algorithm use count
    '''
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    C_all = [] #Holds the currency data for each pool
    for y in poolData:
        C_all.append(y[2:6])
    C = f7(C_all)
    #C = list(set(C_all)) #Make the unique currency data list
    for z in C:
        poolData.append((0,0,z[0],z[1],z[2],z[3])) #appends unique solo mining currencies
        #to poolData where mining power = 0 and fees = 0

    n = len(poolData)
    poolData = [list(l) for l in poolData] #convert to list of lists

    
    minHash = poolData[0][0]
    
    for m in range (0,n-len(C)): #Find smallest value of pool hashrates (exclude solo currencies)
        if (poolData[m][0] < minHash):
            minHash = poolData[m][0]

    for m in range (0,n): 
        poolData[m][0] = poolData[m][0]/minHash #normalize pool hashrates to smallest
        poolData[m][4] = poolData[m][4]/minHash #normalize currency hashrates to smallest
        poolData[m][5] = poolData[m][5]/minHash #normalize miner algo hashrate to smallest
    A_all = [] #Holds the algorithm maximum hashrate for each pool
    for v in poolData:
        A_all.append(v[5]) #Read data from function input
    A_unique = dict.fromkeys(set(A_all),0) #Make the algorithm hashrate dictionary
    for alg in poolData:
        A_unique[alg[5]] += 1 #Populate dictionary with algorithm use count

    x0 = [h[5]/(A_unique[h[5]]+1) for h in poolData] #Initial guess: divide equally per algorithm
    global guess
    guess = x0 #store guess to global variable


    if debug:
        print('Initial Objective: {formatStr}'.format(formatStr=formatStr) % objective(x0))
    #b = (0,max(A_all)) #Use as upper bound the maximum hashrate of all algos, need to check for correctness
    #bnds = (b,) * n
    con1 = {'type': 'ineq', 'fun': constraint1} 
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = ([con1,con2])
    solution = minimize(objective,x0,method='COBYLA',\
                            constraints=cons,options={'maxiter': 50000, 'disp': False, 'tol' : 1e-65, 'catol' : 1e-15})

    #scale back to original values
    x = solution.x
    x = minHash*x


    global curr_num
    curr_num = len(C)

    if debug:
        # show final objective
        print('Final Objective:   {formatStr}'.format(formatStr=formatStr) % objective(x))

        # print solution
        print('Solution')
        curr_ctr = 1
        for n, xi in enumerate(x):
            if (n<len(poolData)-len(C)):
                print('l_%d    = {formatStr}'.format(formatStr=formatStr) % (n, xi))
            else:
                print('Currency',curr_ctr, 'with data:', C[curr_ctr-1])
                print('l_c%d    = {formatStr}'.format(formatStr=formatStr) % (curr_ctr, xi))
                curr_ctr = curr_ctr + 1

    if solution.success:
        return x
    
    return None