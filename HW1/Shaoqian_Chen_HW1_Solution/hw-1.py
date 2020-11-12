import time 
import numpy
import csv

"""
 In this homework, we will be studying two applied topics:
 (1) how to write practical python code!
 (2) how to run meaningful timed experiments by 'simulating' architectures like mapreduce

 This homework is best done after re-starting your machine, and making  sure minimal 'stuff' is open. You may
 need to consult a browser occasionally, as well as the command line, but otherwise please try to minimize
 the programs running  in the background. MAX SCORE POSSIBLE: 100, not including 20 extra credit points.
"""

# let us begin by giving you a crash course in a simple way to measure timing in python. Timing is actually
# a complicated issue in programming, and will vary depending on  your own system. So to make it uniform
# for all students (in procedure, if not in answer), we are telling  you how you should be measuring timing.

# Uncomment the code fragment below, and run it (hint: on  mac, you can select the lines and use command+/.
# Otherwise, look under  the 'Code' menu.

# start = time.time()
# test_list = list()
# for i in range(0,200000000):
#     test_list.append(i)
# end = time.time()
# print(end - start)
# print(len(test_list))
# test_list = None # free up memory

"""
PART I: 15 POINTS
[+5 extra credit points]

Q1. Did the code fragment run without errors? What's the answer? [2+2 points]
ANS: 
    - Error occurred at print len(test_list) => print(len(test_list), due to different version of Python
    - Answer: 0.0, 10

Q2: What's the answer when you change range(0,x) from x=10 to x=1000? What  about x= 100000? [3+3 points]
(I don't care about the 'units' of the timing, as long as you're consistent. However, you should look up
the Python doc and try to be precise about what units the code is outputting i.e. is is milliseconds, seconds, etc.?)
ANS:
    - x=1000: 0.0 seconds, 1000
    - x=100000: 0.007982730865478516 seconds, 100000

Q3: Given that we 'technically' increased the input by 100x (from x=10 to x=1000) followed by another 100x increase, do
you see a similar 'slowdown' in timing? What are the slowdown factors for both of
the 100x input increases above COMPARED to the original input (when x=10)? (For example, if I went from 2 seconds to
30 seconds when  changing from x=10 to x=1000, then the slowdown  factor is 30/2=15 compared to the original input) [5 points]
ANS: 
    - Theoretically the slowdown factor should be approximately 100 times. 
    - Cannot tell from x=10 to x=1000 since both of the time was so small that the console outputs 0.0 seconds for both
    - cannot tell from x=1000 to x=100000 since x=1000 is 0.0 seconds
    - From x=100000 to x=10000000 slowdown factor = 0.9385159015655518/0.007982730865478516 = 117.568

Note 1: If you have a Mac or Unix-based system, a command  like 'top' should work on the command line.
This is a useful command because it gives you a snapshot of threads/processes running on  your machine
and will tell you how much memory is free and being used. See a snapshot in the content folder from
running 'top' on my laptop. You will need to run this command or its equivalent (if 'top' is not applicable to your specific
OS) if  you want the extra credit in some of the sections (not necessary if you don't)

Extra Credit Q1 [5 points]: If you run  top right now, how much memory is free on your machine? What is  the value of x
in range(0,x) for which roughly half of your current free memory gets depleted?

ANS:
    - 13.2 GB
    - Approximately x = 200000000 => 6.0 GB

Note 2: Please re-comment  the code fragment above after you're done with Q1-3. You should be able
to use/write similar timing code to measure timing for the sections below.
"""

"""
Now we are ready to start engaging with numpy. Make sure you have installed it using either pip or within
Pycharm as I showed in class. We will begin with a simple experiment based on sampling. First, let's review
a list of distributions you can sample from here: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html

Uncomment lines 74 and 78 below.
"""

# let's try sampling values from normal distribution:
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html#numpy.random.normal

# seed = numpy.random.seed(1)
# sample = numpy.random.normal(size=1) # The first
# two parameters of the normal function, namely loc and scale have default values specified. Make sure you understand
# this, since it will help you read documentation and learn how to use python libraries.

# print(sample)

"""
PART II: 25 POINTS
[+5 extra credit]
Q1. What is the value output when you run the code above? [3 points]
ANS: 
    - [-0.60557951]

Q2. Run the code five times. Do you get the same value each time? [3 points]
ANS: 
    - No, I got different values each time

Extra Credit Q2 [5 points]: How do I guarantee that I get the same value each time I run the code above (briefly state
your answer below)? Can you add
a command before sample = numpy.random.normal(size=1) such that every time we execute this code we get the same
'sample' value printed? Write such a command above the 'sample = ' command and comment it out.

ANS:
    - Apply the numpy.random.seed(seed=#) function to initialize the RandomState of the machine, using the same #
    seed will provide you with the same RandomState every time you run the program.

Q3: What happens if I run the code with no arguments (i.e. delete size=1)? Does the program crash? Briefly
explain the difference compared to the size=1 case. [9 points]
ANS:
    - With Out Argument = 1.6243453636632417.
    - With Argument = [1.62434536]
    - The program did not crash.
    - Result are the same with the exception of the number of decimal places in each of the answer and one is in an array.
    - With no arguments the code outputs a single value, with the argument of size=1 the output are in the form of an array.

Q4: Add a code fragment below this quoted segment to sample and print 10 values from a laplace distribution
where the peak of the distribution is at position 2.0 and the exponential decay is 0.8. Run the code and paste the
output here. Then comment the code. (Hint: I provided a link to available numpy distributions earlier in this exercise.
Also do not forget to carefully READ the documentation where applicable, it does not take much time) [10 points]
 ANS:
    - [ 1.85482471  2.46478246 -4.70630596  1.59753564  1.01933     0.64868328
  1.21002927  1.70444824  1.81499384  2.06465008]
"""

# paste your response to Q4 below this line.
# seed = numpy.random.seed(1)
# sample = numpy.random.laplace(loc=2.0,scale=0.8,size=10) # The first
# print(sample)


"""
PART III: 60 POINTS
[+10 extra credit]

This is where we start getting into the weeds with simulated mapreduce. As a first step, please make sure to download
words_alpha.txt. It is a list of words. We will do wordcount, but with a twist: we will 'control' for the distribution
of the words to see how mapreduce performs.
"""

# we will first use the numpy.choice function that allows us to randomly choose items from a list of items:
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html#numpy.random.choice

"""
Q1: Read in the words in words_alpha.txt into a list (Hint: do not underestimate this 'simple' task, even
though it is only a few commands; being up close and personal with I/O in any language the first time
can take a couple of trials) Fill in your code in read_in_dictionary_to_list below. Next, right after the function,
 uncomment the print statement. How many words in the list? [10 points].
 ANS:
    - 370103 words in the list

Q2: Now we will write a function to 'generate' a document (in generate_document). It takes a list_of_words (e.g., the output
of read_in_dictionary...) as input and outputs another list of length document_length, which you can basically think about
as a 'document' with document_length (possibly repeated) words.
We can use any probability distribution (over the words in the original input list) to generate such a document. Let's try the
choice function as an obvious choice (no pun intended). Go to the link I provided above to read about it. Now, in
document_generate, write the code to generate a document of LENGTH 20 assuming UNIFORM distribution WITH REPLACEMENT over the original
input list (that you read in from file). (Hint: the tricky part in such generators is setting the value for the 'p' argument, but do you
really need to for this question? See the documentation of choice carefully) [10 points]

Extra Credit Q3 [5 points]: Write an alternate generator function (called generate_document_1) that is the same as above
but the distribution is no longer uniform. Instead, assume that the probability of sampling a word with k characters is
inversely proportional to k (Hint: this will mean creating your own p distribution, make sure it is a valid probability distribution!)

Q3: Now we will run a simple map-reduce function. For context, see the 'master' function. I have already written the
map and reduce functions to make life easier for you. Your primary task will be to run timed experiments. As a first step,
go to the 'master' function and fill in the rest of the code as required. At this point you should be able to run
an experiment (by calling master) and output a file with word counts. [15 points]


Q4: Use the code in the first part of this exercise on 'timing' to plot various curves (you can use Excel or another tool to
plot the curves and only use the code here for collecting 'raw' data) where the length of the document is on the x axis
and varies from 10-100000 in increments of 50, and the y-axis is time units.

<Note: Write additional functions as you see fit to collect/write out the raw data.
Do not modify master for this question; we will use it for grading Q3. For each length (i.e. each x value on your plot) in
 your experiment, you MUST generate at least 10 documents, and then average the data over the ten.>

 Specifically:
  (a) Plot a curve showing how much time the mapper takes as the length of a document increases.
   (b) Plot a curve showing how much time the reducer takes as the length of a document increases
   (hint: review the start-end timing methodology at the beginning of this file, and record 'time' variables where
   appropriate to help you get the data for the plot)
  [25 points]

  EXTRA CREDIT [5 points]: Use 'top' to collect data on memory consumption and give me a single plot where memory
  consumption (in standard units of your choice) is on the y-axis instead of time. Is memory consumption linear
  in the length of the documents? If the program terminates quickly for small-length documents, you can plot the curve
  starting from a 'higher' length like 20,000
"""


def read_in_dictionary_to_list(input_file):
    # this function must return a list. Normally, good software engineering requires you to deal with 'edge cases' such
    # as an empty file, or a file that does not exist, but you can safely assume the input_file will be the path
    # to words_alpha.txt.
    # pass # pass is a placeholder that does nothing at all. It is a syntactic device; remove and replace it with your own code when ready
    word_list = []
    with open('words_alpha.txt', 'r') as reader:
        temp = reader.readlines()
        word_list = [x.replace('\n', '') for x in temp]
    # print(word_list)
    return word_list


# print(len(read_in_dictionary_to_list(input_file="words_alpha.txt")))
# what this hopefully also conveys is that you actually have to
# call a function with inputs to get it to 'do something'. Otherwise, it's just defined and sitting there...waiting.


def generate_document(list_of_words, document_length):
    y = numpy.random.choice(list_of_words, size=document_length, replace=True)
    z = list(y)
    return z;

def generate_document_1(list_of_words, document_length):
    #################################################
    # Create Probability Distribution Array
    #################################################
    # Make array with word lengths of each word
    word_len = [len(i) for i in list_of_words]
    # Find the number of unique word lengths
    num_unique = list(numpy.unique(word_len))
    # Inverse the word lengths for the inverse proportion
    inv_relation_char_len = [1 / i for i in num_unique]
    # Create the probability dist based on inverse word length
    p_dist = []
    for i in range(0, len(word_len), 1):
        for j in range(0, len(num_unique), 1):
            if num_unique[j] == word_len[i]:
                p_dist.append(inv_relation_char_len[j])
    # Normalize distribution to a sum of 1
    p_dist_normalized = p_dist / sum(p_dist)
    ##############################################################
    y = numpy.random.choice(list_of_words, size=document_length, replace=True, p=p_dist_normalized)
    z = list(y)
    return z;

# x = read_in_dictionary_to_list(input_file="words_alpha.txt")
# hello = generate_document_1(x,5)
# print(hello)

# this function is only for instructional purposes, to help you understand dicts. Don't forget to call it to get it to do something!
def playing_with_dicts():
    test_dict = dict()
    # keys and values can be any data type. In this case, the key is string and the value is integer. Values can potentially be
    # complicated data structures, even Python dictionaries themselves!
    sample_words = ['the', 'cow', 'jumped', 'over', 'the', 'moon']
    for s in sample_words:
        if s not in test_dict:  # is s a key already in test_dict?
            test_dict[s] = 0  # initialize count of s to 0
        test_dict[s] += 1  # at this point, s HAS to be a key in test_dict, so we just increment the count
    print(test_dict)  # what happens? make sure to play and experiment so that you get the hang of working with dicts!


# playing_with_dicts() # uncomment for the function above to do something.


"""
Background on global_data (if you don't want to read this, make sure to check out the code fragment at the very end
of this file; it provides some intuition on what I've explained below and gives you a way to try things out!):

Just like in class, the mapper takes a 'document' i.e. a list that you generated
as input. The tricky part is the output, since in an 'implemented' mapreduce system, there is a mechanism to ensure that the 'key-value' pairs that are
emitted by the mapper are routed correctly to the reducer (as we discussed in class, all this means is that the key-value
pairs that have the same key are guaranteed to be routed to the same reducer. Recall that reducers and mappers do not share information
otherwise, thus being embarrassingly parallel). In this implementation, because everything is being done within a single
program we will use a global data structure called 'global_data' to simulate this mechanism. As the code shows, global_data
is a 'dict()' which is a python data structure (DS) that supports keys and values. See my code fragment playing_with_dicts, play with
them! They're the most 'pythonic' DS of all.

So where does global_data come in? We use it to store all key-value pairs emitted by the mapper. To do so, the 'value'
in global_data is actually a list. The list contains all the values emitted by individual mappers. For example, imagine
that the word 'the' occurs thrice in one document and twice in another document. global_data now contains a key 'the'
with value [3,2]. The reducer will independently receive key-value pairs from global_data.

Reduced_data works in a similar way; it records the outputs of the reducer. You will have to write out the outputs
of reduced_data to file.

"""

global_data = dict()
reduced_data = dict()


# already written for you
# as a unit test, try to send in a list (especially with repeated words) and then invoke print_dict
def map(input_list):
    local_counts = dict()
    for i in input_list:
        if i not in local_counts:
            local_counts[i] = 0
        local_counts[i] += 1
    for k, v in local_counts.items():
        if k not in global_data:
            global_data[k] = list()
        global_data[k].append(v)


def print_dict(dictionary):  # helper function that will print dictionary in a nice way
    for k, v in dictionary.items():
        print(k, ' ', v)


# already written for you
def reduce(map_key,
           map_value):  # remember, in python we don't define or fix data types, so the 'types' of these arguments
    # can be anything you want!
    total = 0
    if map_value:
        for m in map_value:
            total += m

    reduced_data[map_key] = total


def master(input_file, output_file):  # see Q3
    word_list = read_in_dictionary_to_list(input_file)

    # write the code below (replace pass) to generate 10 documents, each with the properties in Q2. You can use a
    # list-of-lists i.e. each document is a list, and you could place all 10 documents in an 'outer' list. [6 points]
    # pass
    list_of_doc = []
    for i in range(0, 10, 1):
        list_of_doc.append(generate_document(word_list, 10))

    # Call map over each of your documents (hence, you will have to call map ten times, once over each of your documents.
    # Needless to say, it's good to use a 'for' loop or some other iterator to do this in just a couple of lines of code [ 6 points]
    # pass
    for i in range(0, 10, 1):
        map(list_of_doc[i])

    for k, v in global_data.items():  # at this point global_data has been populated by map. We will iterate over keys and values
        # and call reduce over each key-value pair. Reduce will populate reduced_data
        reduce(k, v)

    # write out reduced_data to output_file. Make sure it's a comma delimited csv, I've provided a sample output in sample_output.csv [3 points]
    with open(output_file, mode='w') as f:
        f_write = csv.writer(f, delimiter=',')
        for key, value in reduced_data.items():
            f_write.writerow([key, value])
    # pass


# master(input_file="words_alpha.txt", output_file="test.csv") # uncomment this to invoke master

# simple test for map/reduce (uncomment and try for yourself!)
# map(['the', 'cow', 'jumped', 'over','the','moon'])
# map(['the', 'sheep', 'jumped', 'over','the','sun'])
# print(global_data)
# for k, v in global_data.items():
#     reduce(k,v)
# print(reduced_data)

## Q4 Function
def question4(input_file, output_file, doc_len):  # see Q3
    word_list = read_in_dictionary_to_list(input_file)

    # write the code below (replace pass) to generate 10 documents, each with the properties in Q2. You can use a
    # list-of-lists i.e. each document is a list, and you could place all 10 documents in an 'outer' list. [6 points]
    # pass
    list_of_doc = []
    for i in range(0, 10, 1):
        list_of_doc.append(generate_document(word_list, doc_len))

    # Call map over each of your documents (hence, you will have to call map ten times, once over each of your documents.
    # Needless to say, it's good to use a 'for' loop or some other iterator to do this in just a couple of lines of code [ 6 points]
    # pass

    map_array = []
    for i in range(0, 10, 1):
        map_start = time.time()

        # Map
        map(list_of_doc[i])

        map_end = time.time()
        map_time = map_end - map_start
        map_array.append(map_time)
    # print('Map Timer: ', map_time)
    sum_map = numpy.sum(map_array)  #Sum up the total time of mapper

    reduce_array = []
    for k, v in global_data.items():  # at this point global_data has been populated by map. We will iterate over keys and values
        # and call reduce over each key-value pair. Reduce will populate reduced_data
        reduce_start = time.time()

        # Reduce
        reduce(k, v)

        reduce_end = time.time()
        reduce_time = reduce_end - reduce_start
        reduce_array.append(reduce_time)
    # print('Reduce Timer: ',reduce_time)
    sum_reduce = numpy.sum(reduce_array)    #Sum up the total time of reducer

    # write out reduced_data to output_file. Make sure it's a comma delimited csv, I've provided a sample output in sample_output.csv [3 points]
    with open(output_file, mode='w') as f:
        f_write = csv.writer(f, delimiter=',')
        for key, value in reduced_data.items():
            f_write.writerow([key, value])
    # pass
    return sum_map, sum_reduce


map_means = []
reduce_means = []
map_val = []
reduce_val = []
j_array = []
for j in range(10, 100000, 50):
    for i in range(0,10,1):        # Generate 10 corpora to get avg map and reduce time
        # Clear Global Data Structures
        global_data = dict()
        reduced_data = dict()
        t_1, t_2 = question4(input_file="words_alpha.txt", output_file="q4.csv", doc_len=j)
        map_means.append(t_1)
        reduce_means.append(t_2)
    # Mean of 10 MapReduce Tasks
    map_val.append(numpy.mean(map_means))
    reduce_val.append(numpy.mean(reduce_means))
    j_array.append(j)
    print(j)
with open('raw_data.csv', mode='w') as f:
    f_write = csv.writer(f, delimiter=',')
    for i in range(0, len(j_array), 1):
        f_write.writerow([j_array[i], map_val[i], reduce_val[i]])