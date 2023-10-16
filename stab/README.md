Replication study of a paper by Stab & Gurveych (2017) on the Hewlett's Foundation's Automated Student Assessment Prize (ASAP) dataset. 

- **Citation**: Stab, C., & Gurevych, I. (2017). Parsing Argumentation Structures in Persuasive Essays. *Computational Linguistics*, 43(3), 619–659. https://doi.org/10.1162/COLI_a_00295


## Datasets  
- Essay Set 1 (ASAP): Effects of computers on people 
- Essay Set 2 (ASAP): Censorship in libraries 
- Annotated Essay Dataset (Stab & Gurveych 2017): https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

## Preprocessing Pipeline 
Unless otherwise specified, we are using the corresponding tools from Stanford CoreNLP. 
1. Tokenization 
2. Segmentation 
    - S&G identified paragraphs by checking for line breaks. There are no line breaks in the ASAP essays, so we will treat every essay as a single paragraph. They used the LanguageTool segmenter 
3. Lemmatization 
    - S&G used the MateTools lemmatizer 
4. Part of Speech tagging 
5. Constituency and Dependency Parsing 
    - Klein and Manning 2003 --> https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/parser/lexparser/LexicalizedParser.html 
6. Sentiment Analysis (both sentence-level and token-level)
    - S&G measure subjectivity by examining the "[p]resence of negative words; number of negative, positive, and neutral words; number of positive words subtracted by the number of negative words." 
7. PDTB (Penn Discourse Treebank) Discourse Parser 
    - Original Ruby implementation by Lin et al.: https://github.com/linziheng/pdtb-parser
    - Java Implementation: https://github.com/WING-NUS/pdtb-parser

## Features for Argument Component Identification 
Structural: 
*See the structural.py file*
- Token Position: 
    - Token present in introduction or conclusion 
    - token is first or last token in sentence
    - relative and absolute token position in document, paragraph and sentence
- Token Punctation: 
    - Token precedes or follows any punctuation, full stop, comma and semicolon
    - token is any punctuation or full stop
- Position of covering sentence: 
    - Absolute and relative position of the token’s covering sentence in the document and paragraph

Probability: 
- Conditional probability of the current token being the beginning of a component (Arg-B) given its n preceding tokens (with n \in {1,2,3}). Use maximum likelihood estimation to estimate these probabilities in the training data. 
- *Preprocessing*: Label each token as Arg-B (beginning of argument component), Arg-I (token within a component) or O (non-argumentative token). The argument components are found in the annotated files from Stab and Gurveych's dataset.  

Syntactic: 
- Part-of-speech
- Lowest common ancestor (LCA): Normalized length of the path to the LCA with the following and preceding token in the parse tree
- LCA types: The two constituent types of the LCA of the current token and its preceding and following token

Lexico-Syntactic: 
- From Stab and Gurveych 2017: "We use lexical head projection rules (Collins 2003) implemented in the Stanford tool suite to lexicalize the constituent parse tree. For each token t, we extract its uppermost node n in the parse tree with the lexical head t and define a lexico- syntactic feature as the combination of t and the constituent type of n. We also consider the child node of n in the path to t and its right sibling, and combine their lexical heads and constituent types as described by Soricut and Marcu (2003)."