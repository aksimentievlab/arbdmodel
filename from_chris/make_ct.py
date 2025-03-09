import numpy as np
import re

name = "aptamer-knotted"
seq = "CAGCACCGACCTTGTGCTTTGGGAGTGCTGGTCCAAGGGCGTTAATGGACA"
dot_bracket_string = "(((((({{.{{{{{..........))))))...}}}}}.}}.........."

dot_bracket = "(((((({{.{{{{{..........))))))...}}}}}.}}.........."
"((((((...((..........)).)))))).....(((((((((((((...+.)))))..))))))))"


def _three_prime_list_to_five_prime(three_prime):
    five_prime = -np.ones(three_prime.shape, dtype=int)
    has_three_prime = np.where(three_prime >= 0)[0]
    five_prime[three_prime[has_three_prime]] = has_three_prime
    return five_prime  

def dot_bracket_to_mrdna_lists(dot_bracket_string):
    ## Find number of nucleotides, neglecting breaks between strands

    split_string = re.split('[+&]',dot_bracket_string)
    strand_end_ids = np.cumsum(np.array( [len(s) for s in split_string], dtype=int ))-1
    num_nt = strand_end_ids[-1] + 1

    # strand_end_ids = np.array([i for i,c in
    #                             zip(range(len(dot_bracket_string)),dot_bracket_string)
    #                             if c in ('+','&')], dtype=int)
    # strand_end_ids = strand_end_ids - np.arange(len(strand_end_ids))

    
    new_string = "".join(split_string)

    ## Set up array indices
    index = np.arange(num_nt,dtype=int)
    three_prime = index+1
    for i in strand_end_ids:
        three_prime[i] = -1
    five_prime = _three_prime_list_to_five_prime(three_prime)
    basepair = -1*np.ones((num_nt), dtype=int)

    nest_level = np.zeros(basepair.shape, dtype=int)
    
    ## Parse new_string for basepairs
    opener_ids = {symbol:[] for symbol in '( { <'.split()}
    closer_to_opener = {close_:open_ for open_,close_ in zip('( { <'.split(),') } >'.split())}

    for i,char in zip(range(num_nt),new_string):
        nest_level[i] = sum([len(opener_ids[k]) for k in opener_ids.keys()])
        if char in opener_ids:
            opener_ids[char].append(i)
        elif char in closer_to_opener:
            o = closer_to_opener[char]
            j = opener_ids[o].pop()
            basepair[i] = j
            basepair[j] = i

    return index, five_prime, three_prime, basepair, nest_level


def lists_to_ct(sequence, index, five_prime, three_prime, basepair, name="fold"):
    num_nt = index[-1]
    ret_string = ["{} {}".format(num_nt,name)]
    
    seq = "".join(re.split('[+&]',sequence))

    for arr in zip(index+1,
                   seq,
                   five_prime+1,
                   three_prime+1,
                   basepair+1,
                   index+1):
        ret_string.append(" ".join([str(i) for i in arr]))
    return "\n".join(ret_string)

"""
java -cp ~cmaffeo2/.local/src/VARNAv3-93.jar fr.orsay.lri.varna.applications.VARNAcmd -i $f -o ${f%ct}png \
    -algorithm radiate \
    -flat true \
    -bpStyle line \
    -spaceBetweenBases 1.0 \
    -resolution 4 \
    -title "" \
    -titleColor '#AAAAAA'  -titleSize 1 \
    -baseNum '#000000'
"""

def dot_bracket_to_ct(sequence, secondary_structure):
    lists = dot_bracket_to_mrdna_lists(secondary_structure)
    return lists_to_ct(sequence,*lists)

if __name__ == '__main__':
    print( dot_bracket_string )
    print( dot_bracket_to_mrdna_lists( dot_bracket_string ) )
    # print( dot_bracket_to_ct( seq, dot_bracket_string ) )



