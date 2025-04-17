import sys
import time

def approxGreedy(n, m, subsets, cutoff):

    '''
    In this algorithm, we essentially are continiously taking the subset with the most new values.
    We keep track of the covered values and the overall values to find out what values are left.
    We end the iteration when covered is equal to the overall values meaning our chosen sets encompass all the values.
    We perform this in a greedy manner since we always take the set with the max new values.
    '''

    #total vals is the complete set of numbers in the input
    totalVals = set(range(1, n + 1))
    covered = set() #empty initial set to keep track of added values
    tot = [] #list of the selected values

    startTime = time.time()

    while covered != totalVals:
        if time.time() - startTime > cutoff:
            break 
        tempInd = -1
        maxVal = -1
        
        #here we are essentially iterating through the subsets and finding the set that has the most new values
        #we find the temp set by removing the already covered values and then take len to find the num of new vals
        for i, s in enumerate(subsets):
            tempSet = s-covered
            newVals = len(tempSet)
            if newVals > maxVal:
                maxVal = newVals
                tempInd = i

        if tempInd == -1:
            break

        #adding values with an OR operator
        covered |= subsets[tempInd]
        #adding index to the return list
        tot.append(tempInd)

    return tot


def parseInput(filename):
    #parses the input file 
    with open(filename, 'r') as file:
        lines = file.readlines()

    #split to find the num vals and number of sets
    n, m = map(int, lines[0].strip().split())
    #creating a list of sets by parsing the input lines
    subsets = [set(map(int, line.strip().split()[1:])) for line in lines[1:]]
    return n, m, subsets


def outputFile(filename, selected_indices):
    #writes to an output file with the given name
    with open(filename, 'w') as f:
        f.write(f"{len(selected_indices)}\n")
        f.write(" ".join(str(i + 1) for i in sorted(selected_indices)))  # 1-based indexing



if __name__ == "__main__":

    inputFileName = sys.argv[1]
    outputFileName = sys.argv[2]
    cutOff = int(sys.argv[3])

    n, m, subsets = parseInput(inputFileName)
    setVals = approxGreedy(n, m, subsets, cutOff)
    outputFile(outputFileName, setVals)

