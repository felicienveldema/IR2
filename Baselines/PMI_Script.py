import sys
from tqdm import tqdm
import nltk
#nltk.download('punkt')
import math
from collections import Counter



def make_Grams(src_file):
    with open(src_file, "r") as sf:
        lines = sf.readlines()
        unigram_freq = Counter()
        bigram_freq = Counter()
        for line in tqdm(lines):
            line = line.lower()
            tokens = nltk.word_tokenize(line)
            bigrams = list(nltk.bigrams(line.split()))
            bigramsC = Counter(bigrams)
            tokensC = Counter(tokens)
            unigram_freq += tokensC
            bigram_freq += bigramsC
    return unigram_freq, bigram_freq






def pmi(word1, word2, unigram_freq, bigram_freq):
    prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
    prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
    prob_word1_word2 = bigram_freq[word1, word2] / float(sum(bigram_freq.values()))

    #unk saves
    if prob_word1 == 0 or prob_word2 == 0:
        return 0
    if prob_word1_word2 == 0:
        return 0

    return math.log(prob_word1_word2/float(prob_word1*prob_word2),2)



def main(args):
    if len(args) < 4 :
        print("not enough input. type 'help' for info")
        sys.exit(0)
    if args[1] == "help" :
        print("Input the source \n Input the 2 words for Pmi")
        sys.exit(0)
    unigrams, bigrams = make_Grams(args[1])

    print(sum(unigrams.values()))
    print(sum(bigrams.values()))

    print("PMI of %s & %s:", args[2], args[3])
    print(pmi(args[2], args[3], unigrams, bigrams))

if __name__ == '__main__':
    main(sys.argv)
