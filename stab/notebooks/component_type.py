'''
Label each component as claim, major claim, or premise 

Use SVM (Cortes and Vapnik 1995) with a polynomial kernel imple- mented in the Weka machine learning framework (Hall et al. 2009)

Features: 
1. Unigrams 
2. Dependency tuples 
3. Token statistics 
4. Component position 
5. Type indicators 
6. First person indicators 
7. Type indicators in context 
8. Shared phrases 
9. Subclauses, depth of parse tree, tense of main verb, modal verbs, POS distrubtion 
10. Type probability: component being one of the three types given its preceding tokens 
11. Discourse triples: PDTB discourse relations overlapping with the current component 
12. Combined word embeddings -- sum of the word vectors of each word of the component and its preceding tokens 
'''