import sys
import os
import time
import argparse
import re
from copy import copy, deepcopy
import pulp
from functools import reduce
import pickle
import multiprocessing
import random
import datetime
import tqdm

import NeuralNetwork
# import matplotlib.pyplot as plt


NEURAL_NETWORKS_N = 20 # must be multiple of 5 !!!
ITERATION_N = 40



#########################################################
# Lire 3 fois le problème
# Communiquer afin de bien être d'accord sur ce qu'il faut faire (le sujet du problème)
# Mettre sujet sur téléphone

# Voir ensemble algo:
#   - complexité
#   - structure de données à utiliser
#   - structure principale du code (étapes importantes)
#   - Performances (travailler avec des index, faire des listes et dictionnaire constant, optimiser la boucle appelée le plus de fois)
#   - les différentes inputs
#   - analogie avec un ancien problème ou autre
#
# Checkpoint:
#   toute les 30 minutes
#
#########################################################
if __name__ == '__main__':

    FILE = "f_libraries_of_the_world"
    DEFAULT_F = "in/" + FILE + ".txt"
    OUTPUT_F = "out/" + FILE + ".txt"
    parser = argparse.ArgumentParser(description='Hashcode 2020 qualif')
    parser.add_argument('-i', '--filein', help='Input file',default=DEFAULT_F, type=str)
    parser.add_argument('-o', '--fileout', help='Ouput file',default=OUTPUT_F, type=str)
    parser.add_argument('-v', '--verbose', default=-1,help='verbose mode', type=int)
    args = parser.parse_args() 



    # map(int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
    # [int(x) for x in re.sub("\n", "", f.readline()).split(" ")]
    # [str(x) for x in re.sub("\n", "", f.readline()).split(" ")]

    
    # Input process
    print("Getting input...")
    with open(args.filein) as f:
        booksN, librarieN, deadline = map(int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
        booksValue = [int(x)for x in re.sub("\n", "", f.readline()).split(" ")]
        booksValue = [(i, booksValue[i]) for i in range(len(booksValue))]
        librairies = []
        for i in range(librarieN):
            (librairiesBooksN, signupProcess, booksPerDay) = map(int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
            books = [int(x) for x in re.sub("\n", "", f.readline()).split(" ")]
            librairies.append([i, librairiesBooksN, signupProcess, booksPerDay, books])
    booksValue2 = sorted(booksValue, key=lambda c: c[1], reverse=True)
    booksIDSortedPerValue = [i[0] for i in booksValue2]

    def priority_sort(to_sort, order):
        priority = {e: p for p, e in enumerate(order)}
        return list(sorted(to_sort, key=lambda e: priority[e]))

    print("Sorting books in librairies....")
    # sort books per value and compute moySignupProcess
    librairieBookScore = []
    moySignupProcess = 0
    pbar = tqdm.tqdm(total=len(librairies))
    # sort book in librairies by values and compute moySignupProcess
    for i, lib in enumerate(librairies):
        lib[4] = priority_sort(lib[4], booksIDSortedPerValue)
        moySignupProcess += lib[2]
        pbar.update(1)
    moySignupProcess = moySignupProcess/len(librairies)
    print(" Average signup process : %f" % (moySignupProcess))


    def write_output(args,best_score,best_conf):
        print("\n")
        print(" Best score: %d"%best_score)
        print("Writting output")
        # output process
        with open(args.fileout, "w") as f:
            f.write(str(len(best_conf))+"\n")
            for l in best_conf:
                f.write(str(l[0])+" "+str(len(l[1]))+"\n")
                for b in l[1]:
                    f.write(str(b)+" ")
                f.write("\n")

    def compute_librairie_book_score(booksValue, lib, time, bookAlreadyAdd):
        # add book until end and sum values
        score = 0 ;i = 0 ;t = 0;n = 0;booksPerDay = lib[3] ;books = [];
        while(i < len(lib[4]) and t < time):
            n  = i
            while(i < min(booksPerDay+n, len(lib[4]))):
                book = lib[4][i]
                if(bookAlreadyAdd[book] == 0):
                    score += booksValue[book][1]
                    books.append(book)
                else:
                    n += 1
                i += 1
            t += 1
        return score, time-t, books

    def get_best_librairies(librairies, booksValue, deadline, timeNow, moySignupProcess, bookAlreadyAdd,neural_network):
        # calculate potential score of all lib
        librairies_scores = []
        for lib in librairies:
            libIndex = lib[0] ; signupTime = lib[2];
            time = deadline-timeNow-1-signupTime
            score, free_day, books = compute_librairie_book_score(booksValue, lib, time, bookAlreadyAdd);
            librairies_scores.append((libIndex, score, free_day, books, signupTime));

        # Apply filters to get the best librairie
        outputs = neural_network.feed_forward([timeNow])
        r1 = int(outputs[0]*len(librairies_scores))+1
        r2 = int(outputs[1]*r1)+1
        r3 = int(outputs[2]*r2)+1
        libSortedScore = librairies_scores
        if(outputs[3]>0.5):
            libSortedScore = sorted(librairies_scores, key=lambda c: c[1],reverse=True)
        if(outputs[4]>0.5):
            libSortedScore = sorted(libSortedScore[0:r1],key=lambda c: c[4])
        if(outputs[5]>0.5):
            libSortedScore = sorted(libSortedScore[0:r2],key=lambda c: c[2])
        # sort by other params
        if(r3<1):
            return libSortedScore[0][0], libSortedScore[0][3]
        r = random.randint(0,r3-1)
        return libSortedScore[r3-1][0], libSortedScore[r3-1][3]



    def solve(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies,neural_network):
        # while he don't exceeds the max time we search the best librairies and we add it
        timeNow = 0;   bookAlreadyAdd = {i: 0 for i in range(booksN)}; lib_score = 0;final = [];
        while(timeNow < deadline-1 and len(librairies) > 0):
            # get best lib
            next_lib, books = get_best_librairies(librairies, booksValue, deadline, timeNow, moySignupProcess, bookAlreadyAdd,neural_network)
            # delete from the database the lib that we just add
            for i, l in enumerate(librairies):
                if l[0] == next_lib:
                    next_lib = deepcopy(l)
                    break
            del librairies[i]
            # increment time with the librairie's signup time
            timeNow += next_lib[2]
            # if their is book we can add the lib in  the final solution
            if(len(books) > 0):
                final.append((next_lib[0], books))
            # update other book
            for book in books:
                bookAlreadyAdd[book] = 1
                lib_score += booksValue[book][1]
        return lib_score,final


    def test_neural_networks(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies,neural_networks,pbar):
        results = []
        # compute score of each neural_networks
        for i in range(len(neural_networks)):
            libs= deepcopy(librairies)
            score,final = solve(booksN, librarieN, deadline, booksValue,moySignupProcess, libs, neural_networks[i])
            results.append((neural_networks[i],score,final))
        return results


    def cross(n1,n2):
        return n1.cross(n2)

    def mutate(n1):
        return n1.mutate()


    def cross_and_mutate(results):
        new_neural_network  =[]
        N= len(results)
        for i in range(2*N//5):
            new_neural_network.append(deepcopy(results[i][0]))
        for i in range(1*N//5):
            new_neural_network.append(deepcopy(results[i][0].cross(results[i+1][0])))
        for i in range(N//5):
             new_neural_network.append(deepcopy(mutate(results[i][0].cross(results[i+1][0]))))
        for i in range(N//5):
            new_neural_network.append(NeuralNetwork.NeuralNetwork(1,40,6))
        return new_neural_network
        

    def train(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies,N,M):
        # main loop here
        # create, cross, mutate and get best score ITERATION_N times !
        neural_networks = []
        best_score = 0; best_conf = False;
        # create neural networks
        for i in range(M):
            neural_networks.append(NeuralNetwork.NeuralNetwork(1,40,6))
        pbar = tqdm.tqdm(total=N)
        print(" Starting training...")
        # Run iterations
        for i in range(N):
            results = test_neural_networks(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies,neural_networks,pbar)
            results = sorted(results, key=lambda c: c[1],reverse=True)
            neural_networks = cross_and_mutate(results)
            pbar.update(1)
            pbar.set_postfix(score=results[0][1],l=len(neural_networks))
            # save best score
            if(results[0][1]> best_score):
                best_score = results[0][1]
                best_conf = results[0][2]
        # cross, mutate, destroy and create
        return best_score,best_conf



best_score , best_conf = train(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies,ITERATION_N,NEURAL_NETWORKS_N)
write_output(args,best_score,best_conf)


 

    


    