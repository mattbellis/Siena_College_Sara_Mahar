#random function to find random answers to put on a multiple choice question

import random
import numpy as np

#def random_generator(correct_answer,answer_range,number_of_answers):

correct_answer=503
answer_range=800
number_of_answers=10

answers=1e9*np.ones(number_of_answers)

correct_answer_location = random.randint(0,(number_of_answers-1))

max_answer = correct_answer + (1 - correct_answer_location/float(number_of_answers))*answer_range
min_answer = correct_answer - (    correct_answer_location/float(number_of_answers))*answer_range
#min_answer = correct_answer - random.random()*answer_range

print correct_answer_location
print min_answer,correct_answer,max_answer

#exit()


correct_answer_index=0

answers[correct_answer_index] = correct_answer

#print "index for the CORRECT answer: %d " % correct_answer_index

for num in range(1,number_of_answers):
    #print "At %d in Number of answers: %d" % (num,number_of_answers)

    found_possible_answer= False

    while found_possible_answer==False:

         rand_num=random.uniform(min_answer,max_answer)
         
	 for check_index in range(0,num):

              #print "checking the answers: %.4f with random number: %.4f" % (answers[check_index], rand_num)
              fraction_difference = (rand_num - answers[check_index])/answers[check_index]
              within_tolerance = np.abs(fraction_difference)<0.05
              
	      #print "This is the fraction difference: %.4f" %fraction_difference
              #print within_tolerance

              if within_tolerance==True:
                  break    

         if within_tolerance==False:
             answers[num] = rand_num
             found_possible_answer = True


         #print answers

answers.sort()
rounded_answers = ['%.2f' % elem for elem in answers]
print rounded_answers
