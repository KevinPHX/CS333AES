'''
Sequence labeling task at token level: 
    Encode the argument components using an IOB-tagset (Ramshaw and Marcus 1995) and consider an entire essay as a single sequence. 
    Accordingly, we label the first token of each argument component as “Arg-B”, 
                            the tokens covered by an argument component as “Arg-I”, 
                            and non-argumentative tokens as “O”. 
    Learner: CRF with the averaged perceptron training method (Collins 2002)

Pretty straightforward: 
1. Token position -- if token in intro or conclusion; first or last in a sentence; relative and absolute position 
2. Punctuation -- if token precedes or follows any punctuation or is a puncutation mark 
3. Position of covering sentence -- position of the sentence in which token is situated 
4. Part of speech


More involved: 
5. Lowest common ancestor -- normalized length of the path to the LCA with the following and preceding token in the parse tree 
6. LCA types -- two constituent types of the LCA of the current token and its preceding & following token 
7. Lexico-syntactic: Soricut and Marcu (2003)
8. Probability: Conditional probability of the current token being the beginning of a component given its preceding tokens (maximum likelihood estimation)
'''