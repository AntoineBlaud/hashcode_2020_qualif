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
# import matplotlib.pyplot as plt



#########################################################
# Lire 3 fois le problème
# Communiquer afin de bien être d'accord sur ce qu'il faut faire (le sujet du problème)
# Mettre sujet sur téléphone

# Voir ensemble algo:
#   - complexité
#   - structure de données à utiliser
#   - structure principale du code (étapes importantes)
#   - Performances (travailler avec des index, faire des listes et dictionnaire constant, optimiser la boucle appelée le plus de fois)
#   - les différentes input
#   - analogie avec un ancien problème ou autre
#
# Checkpoint:
#   toute les 30 minutes
#
#
#########################################################
if __name__ == '__main__':

    
    FILE = "e_so_many_books"
    DEFAULT_F = "C:\\Users\\antoi\\Documents\\GitHub\\hashcode\\2020-qualif\\in\\" + FILE + ".txt"
    OUTPUT_F = "C:\\Users\\antoi\\Documents\\GitHub\\hashcode\\2020-qualif\\out\\" + FILE + ".txt"
    parser = argparse.ArgumentParser(description='Hashcode 2020 qualif')
    parser.add_argument('-i', '--filein', help='Input file',
                        default=DEFAULT_F, type=str)
    parser.add_argument('-o', '--fileout', help='Ouput file',
                        default=OUTPUT_F, type=str)
    parser.add_argument('-v', '--verbose', default=-1,
                        help='verbose mode', type=int)
    args = parser.parse_args() 



    # map(int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
    # [int(x) for x in re.sub("\n", "", f.readline()).split(" ")]
    # [str(x) for x in re.sub("\n", "", f.readline()).split(" ")]

    # Input process
    print("Getting input...")
    with open(args.filein) as f:
        booksN, librarieN, deadline = map(
            int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
        booksValue = [int(x)
                      for x in re.sub("\n", "", f.readline()).split(" ")]
        booksValue = [(i, booksValue[i]) for i in range(len(booksValue))]
        

        librairies = []
        for i in range(librarieN):
            (librairiesBooksN, signupProcess, booksPerDay) = map(
                int, [int(x) for x in re.sub("\n", "", f.readline()).split(" ")])
            books = [int(x) for x in re.sub("\n", "", f.readline()).split(" ")]
            librairies.append(
                [i, librairiesBooksN, signupProcess, booksPerDay, books])

    booksValue2 = sorted(booksValue, key=lambda c: c[1], reverse=True)
    booksIDSortedPerValue = [i[0] for i in booksValue2]

    def priority_sort(to_sort, order):
        priority = {e: p for p, e in enumerate(order)}
        return list(sorted(to_sort, key=lambda e: priority[e]))

    # A refaire
    def compute_librairie_book_score(booksValue, lib, time, bookAlreadyAdd):
        score = 0
        i = 0
        t = 0
        n = 0
        booksPerDay = lib[3]
        books = []
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

    def get_Best_librairies(librairies, booksValue, deadline, timeNow, moySignupProcess, bookAlreadyAdd):
        # sort by score
        librairies_scores = []
        for lib in librairies:
            libIndex = lib[0]
            signupTime = lib[2]
            time = deadline-timeNow-1-signupTime
            score, free_day, books = compute_librairie_book_score(booksValue, lib, time, bookAlreadyAdd);
            librairies_scores.append((libIndex, score, free_day, books, signupTime));

        r1 = random.randint(300,700)
        r2 = random.randint(11,30)
        r3 = random.randint(1,10)
        libSortedScore = sorted(librairies_scores, key=lambda c: c[1],reverse=True)
        maxI = min(r1,len(libSortedScore))
        libSortedScore = sorted(libSortedScore[0:maxI],key=lambda c: c[4])
        maxI2 = min(r2,len(libSortedScore))
        libSortedScore = sorted(libSortedScore[0:maxI2],key=lambda c: c[2])
        # sort by other params
        maxI3 =  random.randint(0,r3)
        return libSortedScore[maxI3][0], libSortedScore[maxI3][3]

    def solve(booksN, librarieN, deadline, booksValue,moySignupProcess, librairies):
        print("Starting solving....")
        best_score = 0
        best_conf = []
        REP = 10
        saved_libraires = deepcopy(librairies)
        pbar = tqdm.tqdm(total=REP)
        for test in range(REP):
            lib_score = 0
            timeNow = 0
            final = []
            bookAlreadyAdd = {i: 0 for i in range(booksN)}
            librairies = deepcopy(saved_libraires)
            while(timeNow < deadline-1 and len(librairies) > 0):
                next_lib, books = get_Best_librairies(librairies, booksValue, deadline, timeNow, moySignupProcess, bookAlreadyAdd)
                # supprime la librairie ajouté
                for i, l in enumerate(librairies):
                    if l[0] == next_lib:
                        next_lib = deepcopy(l)
                        break
                del librairies[i]

                timeNow += next_lib[2]
                if(len(books) > 0):
                    final.append((next_lib[0], books))
                    pbar.set_postfix(file=FILE, score=best_score,lib=len(librairies))
                # mettre a jour les autres livres
                for book in books:
                    bookAlreadyAdd[book] = 1
                    lib_score += booksValue[book][1]
            if(lib_score > best_score):
                best_score = lib_score
                best_conf = deepcopy(final)
            pbar.update(1)
            
        return best_conf,best_score

    print("Sorting books in librairies....")
    # sort books per value and compute moySignupProcess
    librairieBookScore = []
    moySignupProcess = 0
    pbar = tqdm.tqdm(total=len(librairies))
    # livre dans les bibliothèques triées par valeur
    for i, lib in enumerate(librairies):
        lib[4] = priority_sort(lib[4], booksIDSortedPerValue)
        moySignupProcess += lib[2]
        pbar.update(1)
    moySignupProcess = moySignupProcess/len(librairies)
    print(" Average signup process : %f" % (moySignupProcess))

    best_conf,best_score = solve(booksN, librarieN, deadline, booksValue, moySignupProcess, librairies)

    print("\n")
    print("Best score: %d"%best_score)
    print(" Writting output")

    # output process
    with open(args.fileout, "w") as f:
        f.write(str(len(best_conf))+"\n")
        for l in best_conf:
            f.write(str(l[0])+" "+str(len(l[1]))+"\n")
            for b in l[1]:
                f.write(str(b)+" ")
            f.write("\n")
